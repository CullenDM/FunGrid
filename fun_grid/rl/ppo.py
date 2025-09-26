"""Proximal Policy Optimization utilities and transition storage for FunGrid."""

from __future__ import annotations

import gc
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config


TransitionType = Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    int,
    int,
    float,
    float,
    float,
    torch.Tensor,
    bool,
]


class TransitionStorage:
    """Transition container that supports proper bootstrapping."""

    def __init__(self) -> None:
        self.transitions: List[Tuple] = []

    def store_transition(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        action: int,
        direction: int,
        action_log_prob: float,
        direction_log_prob: float,
        reward: float,
        value: torch.Tensor | float,
        done: bool,
        truncated: bool = False,
        bootstrap_value: torch.Tensor | float | None = None,
    ) -> None:
        """Store one transition and eagerly move tensors to CPU."""
        view_seq, state_vec = state
        view_cpu = view_seq.detach().cpu()
        state_cpu = state_vec.detach().cpu()
        value_cpu = (
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=torch.float32)
        )

        if truncated and bootstrap_value is None:
            raise ValueError("If truncated=True, bootstrap_value (V(s_{t+1})) is required.")

        if bootstrap_value is not None:
            bootstrap_cpu = (
                bootstrap_value.detach().cpu()
                if isinstance(bootstrap_value, torch.Tensor)
                else torch.tensor(bootstrap_value, dtype=torch.float32)
            )
            self.transitions.append(
                (
                    (view_cpu, state_cpu),
                    action,
                    direction,
                    action_log_prob,
                    direction_log_prob,
                    reward,
                    value_cpu,
                    bool(done),
                    True,
                    bootstrap_cpu,
                )
            )
        else:
            self.transitions.append(
                (
                    (view_cpu, state_cpu),
                    action,
                    direction,
                    action_log_prob,
                    direction_log_prob,
                    reward,
                    value_cpu,
                    bool(done),
                )
            )

    def mark_last_as_truncated(self, bootstrap_value: torch.Tensor | float) -> None:
        """Upgrade the most recent transition to mark it truncated."""
        if not self.transitions:
            return
        last = self.transitions[-1]
        done_flag = bool(last[7])
        if done_flag:
            return

        bootstrap_cpu = (
            bootstrap_value.detach().cpu()
            if isinstance(bootstrap_value, torch.Tensor)
            else torch.tensor(bootstrap_value, dtype=torch.float32)
        )

        if len(last) == 8:
            rebuilt = last + (True, bootstrap_cpu)
        else:
            rebuilt = last[:9] + (bootstrap_cpu,)
        self.transitions[-1] = rebuilt

    def clear_memory(self) -> None:
        """Clear stored transitions and release cached CUDA memory."""
        self.transitions.clear()
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class LossComponents:
    """Container for loss values produced during PPO updates."""

    action_loss: float
    direction_loss: float
    value_loss: float
    compute_time: float


class PPO:
    """Proximal Policy Optimization implementation with optional halting terms."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_param: float = 0.2,
        ppo_epochs: int = 1,
        target_kl: float = 0.01,
        gamma: float = 0.99,
        tau: float = 0.95,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.target_kl = target_kl
        self.gamma = gamma
        self.tau = tau
        self.transitions: List[Tuple] = []

        self.halt_coef = float(getattr(Config, "PPO_HALTING_COEF", 0.0))
        self.ponder_penalty = float(getattr(Config, "PPO_PONDER_PENALTY", 0.0))
        self.sel_ent_coef = float(getattr(Config, "PPO_SELECTOR_ENT_COEF", 0.0))

        self._last_aux: Dict[str, float] = {}

    # The remaining methods implement GAE, optional halting auxiliaries, and PPO updates.
    # They map closely to the original monolithic implementation but have been
    # refactored for modular use within the package structure.

    def compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(Config.DEVICE)
        gamma, tau = self.gamma, self.tau

        T = len(self.transitions)
        if T == 0:
            empty = torch.empty(0, dtype=torch.float32, device=device)
            return empty, empty

        rewards: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        truncs: List[torch.Tensor] = []
        boots: List[torch.Tensor] = []

        for item in self.transitions:
            r = torch.as_tensor(item[5], dtype=torch.float32, device=device)
            v = item[6]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            v = v.to(device).reshape(())
            d = torch.as_tensor(item[7], dtype=torch.float32, device=device)

            if len(item) >= 10:
                tflag = torch.as_tensor(
                    1.0 if item[8] else 0.0, dtype=torch.float32, device=device
                )
                b = item[9]
                if not isinstance(b, torch.Tensor):
                    b = torch.tensor(b, dtype=torch.float32)
                b = b.to(device).reshape(())
            else:
                tflag = torch.tensor(0.0, dtype=torch.float32, device=device)
                b = torch.tensor(0.0, dtype=torch.float32, device=device)

            rewards.append(r)
            values.append(v)
            dones.append(d)
            truncs.append(tflag)
            boots.append(b)

        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        dones_tensor = torch.stack(dones)
        truncs_tensor = torch.stack(truncs)
        boots_tensor = torch.stack(boots)

        returns_all: List[torch.Tensor] = []
        advantages_all: List[torch.Tensor] = []

        start = 0
        while start < T:
            end = start
            while (
                end < T - 1
                and dones_tensor[end].item() == 0.0
                and truncs_tensor[end].item() == 0.0
            ):
                end += 1

            seg_rewards = rewards_tensor[start : end + 1]
            seg_values = values_tensor[start : end + 1]
            seg_dones = dones_tensor[start : end + 1]
            seg_trunc = truncs_tensor[start : end + 1]
            seg_len = seg_rewards.size(0)

            if seg_dones[-1] > 0.5:
                bootstrap = torch.tensor(0.0, dtype=torch.float32, device=device)
            elif seg_trunc[-1] > 0.5:
                bootstrap = boots_tensor[start + seg_len - 1].detach()
            else:
                bootstrap = torch.tensor(0.0, dtype=torch.float32, device=device)

            values_ext = torch.cat([seg_values, bootstrap.view(1)], dim=0)

            gae = torch.tensor(0.0, dtype=torch.float32, device=device)
            seg_returns: List[torch.Tensor] = []
            for i in reversed(range(seg_len)):
                mask = 1.0 - seg_dones[i]
                delta = (
                    seg_rewards[i]
                    + gamma * values_ext[i + 1] * mask
                    - values_ext[i]
                )
                gae = delta + gamma * tau * mask * gae
                seg_returns.insert(0, gae + values_ext[i])

            seg_returns_tensor = torch.stack(seg_returns)
            seg_advantage = seg_returns_tensor - seg_values

            returns_all.append(seg_returns_tensor)
            advantages_all.append(seg_advantage)

            start = end + 1

        returns_tensor = torch.cat(returns_all, dim=0)
        advantages = torch.cat(advantages_all, dim=0)

        mean = advantages.mean()
        std = advantages.std(unbiased=False)
        if (not torch.isfinite(std)) or (std < 1e-8):
            advantages = advantages - mean
        else:
            advantages = (advantages - mean) / std

        returns_tensor = torch.nan_to_num(returns_tensor)
        advantages = torch.nan_to_num(advantages)

        return returns_tensor.detach(), advantages.detach()

    def _aux_halting_terms(
        self, grid_inputs: torch.Tensor, state_inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor] | None:
        if not hasattr(self.model, "get_halting_aux"):
            return None
        aux = self.model.get_halting_aux((grid_inputs, state_inputs))
        if not isinstance(aux, dict) or len(aux) == 0:
            return None
        out: Dict[str, torch.Tensor] = {}
        for key, value in aux.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(Config.DEVICE)
        return out

    def compute_loss(self, minibatch: Dict[str, torch.Tensor]):
        new_a_log, new_d_log, new_vals, entropy = self.model.get_new_log_probs(
            (minibatch["grid_inputs"], minibatch["state_inputs"]),
            minibatch["actions"],
            minibatch["directions"],
        )
        ratio_a = torch.exp(new_a_log - minibatch["old_action_log_probs"])
        surr1 = ratio_a * minibatch["advantages"]
        surr2 = torch.clamp(ratio_a, 1 - self.clip_param, 1 + self.clip_param) * minibatch[
            "advantages"
        ]
        action_loss = -torch.min(surr1, surr2).mean()

        ratio_d = torch.exp(new_d_log - minibatch["old_direction_log_probs"])
        surr1_d = ratio_d * minibatch["advantages"]
        surr2_d = torch.clamp(ratio_d, 1 - self.clip_param, 1 + self.clip_param) * minibatch[
            "advantages"
        ]
        direction_loss = -torch.min(surr1_d, surr2_d).mean()

        v_clipped = minibatch["values"] + torch.clamp(
            new_vals - minibatch["values"], -self.clip_param, self.clip_param
        )
        v_loss_clip = F.smooth_l1_loss(v_clipped, minibatch["returns"])
        v_loss_plain = F.smooth_l1_loss(new_vals, minibatch["returns"])
        value_loss = torch.max(v_loss_plain, v_loss_clip)

        aux_loss = torch.tensor(0.0, device=Config.DEVICE)
        aux_info = {"halt_bce": 0.0, "ponder": 0.0, "sel_entropy": 0.0}

        if any(
            coef > 0.0 for coef in (self.halt_coef, self.ponder_penalty, abs(self.sel_ent_coef))
        ):
            aux = self._aux_halting_terms(minibatch["grid_inputs"], minibatch["state_inputs"])
            if aux is not None:
                if self.halt_coef > 0.0 and "halt_prob" in aux:
                    halt_prob = aux["halt_prob"].clamp(1e-6, 1 - 1e-6)
                    target = torch.ones_like(halt_prob)
                    halt_bce = F.binary_cross_entropy(halt_prob, target)
                    aux_loss = aux_loss + self.halt_coef * halt_bce
                    aux_info["halt_bce"] = float(halt_bce.detach().item())

                if self.ponder_penalty > 0.0 and "segments_used" in aux:
                    max_segs = max(int(getattr(Config, "MODEL_MAX_SEGMENTS", 1)), 1)
                    used = aux["segments_used"].float()
                    if max_segs > 1:
                        norm_used = (used - 1.0) / (max_segs - 1.0)
                    else:
                        norm_used = used * 0.0
                    ponder = norm_used.mean()
                    aux_loss = aux_loss + self.ponder_penalty * ponder
                    aux_info["ponder"] = float(ponder.detach().item())

                if self.sel_ent_coef != 0.0 and "selector_weights" in aux:
                    weights = aux["selector_weights"].clamp_min(1e-12)
                    sel_ent = -(weights * weights.log()).sum(dim=-1).mean()
                    aux_loss = aux_loss - self.sel_ent_coef * sel_ent
                    aux_info["sel_entropy"] = float(sel_ent.detach().item())

        total_loss = (
            action_loss
            + direction_loss
            + value_loss
            - entropy * Config.ENTROPY_COEFFICIENT
            + aux_loss
        )
        self._last_aux = aux_info
        return (action_loss, direction_loss, value_loss), entropy, aux_loss, aux_info

    def update(self, transitions: Sequence[Tuple], minibatch_size: int | None = None) -> LossComponents:
        self.transitions = list(transitions)
        returns, advantages = self.compute_returns_and_advantages()

        if not minibatch_size:
            minibatch_size = getattr(Config, "MINIBATCH_SIZE", len(self.transitions) or 1)

        states, actions, directions, old_a_logp, old_d_logp, _, values, _ = zip(
            *[t[:8] for t in self.transitions]
        )
        grid_in, state_in = zip(*states)
        data = {
            "grid_inputs": torch.stack(grid_in).to(Config.DEVICE),
            "state_inputs": torch.stack(state_in).to(Config.DEVICE),
            "actions": torch.tensor(actions, dtype=torch.int64, device=Config.DEVICE),
            "directions": torch.tensor(directions, dtype=torch.int64, device=Config.DEVICE),
            "old_action_log_probs": torch.tensor(
                old_a_logp, dtype=torch.float32, device=Config.DEVICE
            ).detach(),
            "old_direction_log_probs": torch.tensor(
                old_d_logp, dtype=torch.float32, device=Config.DEVICE
            ).detach(),
            "values": torch.stack(values).to(Config.DEVICE).detach(),
            "returns": returns.to(Config.DEVICE),
            "advantages": advantages.to(Config.DEVICE),
        }

        action_losses: List[float] = []
        direction_losses: List[float] = []
        value_losses: List[float] = []

        if torch.cuda.is_available() and str(Config.DEVICE).startswith("cuda"):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            use_cuda_events = True
        else:
            start_time = end_time = None
            t0 = time.time()
            use_cuda_events = False

        N = data["actions"].size(0)
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(N, device=Config.DEVICE)
            self.optimizer.zero_grad()

            for start in range(0, N, minibatch_size):
                end = min(start + minibatch_size, N)
                mb_idx = indices[start:end]
                minibatch = {key: value[mb_idx] for key, value in data.items()}
                (a_loss, d_loss, v_loss), entropy, aux_loss, _ = self.compute_loss(minibatch)
                total = a_loss + d_loss + v_loss - entropy * Config.ENTROPY_COEFFICIENT + aux_loss
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(getattr(Config, "MODEL_GRAD_CLIP_NORM", 1.0)),
                )

                action_losses.append(a_loss.item())
                direction_losses.append(d_loss.item())
                value_losses.append(v_loss.item())

            self.optimizer.step()

            if getattr(Config, "USE_EARLY_STOPPING", False):
                total_kl = 0.0
                count = 0
                for start in range(0, N, minibatch_size):
                    end = min(start + minibatch_size, N)
                    with torch.no_grad():
                        new_a_logp, _, _, _ = self.model.get_new_log_probs(
                            (
                                data["grid_inputs"][start:end],
                                data["state_inputs"][start:end],
                            ),
                            data["actions"][start:end],
                            data["directions"][start:end],
                        )
                    old_logp = data["old_action_log_probs"][start:end]
                    chunk_kl = (new_a_logp - old_logp).mean().item()
                    total_kl += chunk_kl * (end - start)
                    count += end - start
                kl = total_kl / max(count, 1)
                if kl > self.target_kl:
                    print(f"Early stopping at epoch {epoch + 1}, KL {kl:.6f}")
                    break

        if use_cuda_events and start_time and end_time:
            end_time.record()
            torch.cuda.synchronize()
            compute_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            compute_time = time.time() - t0

        del data, grid_in, state_in, returns, advantages
        torch.cuda.empty_cache()

        return LossComponents(
            action_loss=float(np.mean(action_losses)) if action_losses else 0.0,
            direction_loss=float(np.mean(direction_losses)) if direction_losses else 0.0,
            value_loss=float(np.mean(value_losses)) if value_losses else 0.0,
            compute_time=float(compute_time),
        )

    def clear_memory(self) -> None:
        self.transitions = []

    def save_transitions(self, file_path: str | None = None) -> None:
        if not getattr(Config, "SAVE_TRANSITIONS", False):
            return
        path = file_path or getattr(Config, "TRANSITIONS_FILE_PATH", "transitions.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self.transitions, handle)

