"""Neural network policy definitions and wrappers for PPO training."""

from __future__ import annotations

import importlib
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..config import Config


def _cfg(name: str, default: float | int | str) -> float | int | str:
    """Lookup helper that falls back to a default if Config lacks an attribute."""

    return getattr(Config, name, default)


class RMSNorm1d(nn.Module):
    """Root-mean-square normalisation for 1D feature vectors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math op
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x * (self.g / rms)


class MLP(nn.Module):
    """Two-layer perceptron with configurable activation and dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        act: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoordPointerGRU(nn.Module):
    """Default multi-head PPO policy with pointer-style attention over grid history."""

    def __init__(self) -> None:
        super().__init__()
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.grid_size = Config.GRID_SIZE
        self.hw = self.grid_size * self.grid_size
        self.state_dim = Config.STATE_DIM

        self.num_cell_types = int(_cfg("MODEL_MAX_CELL_ID", 31)) + 1
        self.food_id = int(_cfg("MODEL_FOOD_ID", 10))
        self.agent_id = int(_cfg("MODEL_AGENT_ID", 11))
        self.empty_id = int(_cfg("MODEL_EMPTY_ID", 9))
        self.moveable_id = int(_cfg("MODEL_MOVEABLE_ID", 7))
        self.grabbable_id = int(_cfg("MODEL_GRABBABLE_ID", 8))
        self.v_max = float(_cfg("MODEL_VMAX_VALUE", 1000.0))

        self.selector_tau = float(_cfg("MODEL_SELECTOR_TAU", 0.6))
        self.selector_recency = float(_cfg("MODEL_SELECTOR_RECENCY", 0.3))

        self.ray_steps = int(_cfg("MODEL_RAY_MAX_STEPS", 6))
        self.ray_alpha = float(_cfg("MODEL_RAY_ALPHA", 1.0))
        self.ray_beta = float(_cfg("MODEL_RAY_BETA", 0.5))

        cell_emb_dim = int(_cfg("MODEL_CELL_EMB_DIM", 32))
        step_dim = int(_cfg("MODEL_STEP_DIM", 64))
        gru_hidden = int(_cfg("MODEL_GRU_HIDDEN", 96))
        head_hidden = int(_cfg("MODEL_HEAD_HIDDEN", 96))
        dropout = float(_cfg("MODEL_DROPOUT", 0.0))

        self.dir_bias_scale = float(_cfg("MODEL_DIR_BIAS_SCALE", 1.0))
        self.affordance_scale = float(_cfg("MODEL_AFFORDANCE_SCALE", 0.5))
        self.action_bias_scale = float(_cfg("MODEL_ACTION_BIAS_SCALE", 0.5))

        coords = self._make_xy_coords(self.grid_size, self.grid_size, device=Config.DEVICE)
        self.register_buffer("cell_xy", coords, persistent=False)
        self.center_xy = torch.tensor(
            [self.grid_size // 2, self.grid_size // 2],
            dtype=torch.float32,
            device=Config.DEVICE,
        ) / (self.grid_size - 1.0)

        self.cell_embed = nn.Embedding(self.num_cell_types, cell_emb_dim)
        self.xy_proj = nn.Linear(2, cell_emb_dim, bias=False)
        self.pass_conv = nn.utils.weight_norm(nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, groups=1))

        self.percell_mixer = nn.Sequential(
            nn.Linear(cell_emb_dim * 2 + 2, cell_emb_dim),
            nn.SiLU(),
            nn.Linear(cell_emb_dim, cell_emb_dim),
            nn.SiLU(),
        )

        self.att_all = nn.Sequential(nn.Linear(cell_emb_dim, cell_emb_dim), nn.Tanh(), nn.Linear(cell_emb_dim, 1))
        self.att_food = nn.Sequential(nn.Linear(cell_emb_dim, cell_emb_dim), nn.Tanh(), nn.Linear(cell_emb_dim, 1))
        self.grid_step_proj = nn.Sequential(nn.Linear(cell_emb_dim, step_dim), nn.SiLU())

        self.state_norm = RMSNorm1d(self.state_dim)
        self.grid_norm = RMSNorm1d(step_dim)
        self.fuse_gate = nn.Linear(step_dim + self.state_dim + 3, step_dim)
        self.fuse_proj = nn.Linear(step_dim + self.state_dim + 3, step_dim)

        self.gru = nn.GRU(input_size=step_dim, hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self.post_gru = RMSNorm1d(gru_hidden)

        self.sel_actor_h = nn.Linear(gru_hidden, 64)
        self.sel_actor_s = nn.Linear(self.state_dim, 64)
        self.sel_actor_v = nn.Linear(64, 1)

        self.sel_critic_h = nn.Linear(gru_hidden, 32)
        self.sel_critic_s = nn.Linear(self.state_dim, 32)
        self.sel_critic_v = nn.Linear(32, 1)

        self.pi_action = MLP(gru_hidden * 2, head_hidden, Config.NUM_ACTIONS, dropout=dropout)
        self.pi_direction = MLP(gru_hidden * 2, head_hidden, Config.NUM_DIRECTIONS, dropout=dropout)

        self.critic_pre = nn.Sequential(
            nn.Dropout(p=min(0.1, dropout + 0.05)),
            nn.Linear(gru_hidden, head_hidden),
            nn.SiLU(),
        )
        self.v_head_raw = nn.Linear(head_hidden, 1)

        self.affordance = nn.Embedding(self.num_cell_types, 4)
        self.gate_dir = nn.Linear(gru_hidden * 2, 4)
        self.gate_act = nn.Linear(gru_hidden * 2, Config.NUM_ACTIONS)

        nn.init.constant_(self.gate_dir.bias, 1.0)
        nn.init.constant_(self.gate_act.bias, 1.0)

        self._last_selector_weights: torch.Tensor | None = None

    @staticmethod
    def _make_xy_coords(height: int, width: int, device: str | torch.device) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        xs = xs / (width - 1.0)
        ys = ys / (height - 1.0)
        return torch.stack([xs, ys], dim=-1).view(height * width, 2)

    def _percell_features(self, grid_ids_long: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, hw = grid_ids_long.shape
        emb_dim = self.cell_embed.embedding_dim

        cell_emb = self.cell_embed(grid_ids_long)
        xy = self.xy_proj(self.cell_xy).unsqueeze(0).unsqueeze(0).expand(batch, seq, hw, emb_dim)

        is_food = (grid_ids_long == self.food_id).float()
        passable = (
            (grid_ids_long == self.empty_id)
            | (grid_ids_long == self.food_id)
            | (grid_ids_long == self.moveable_id)
        ).float()

        pass_density = torch.sigmoid(self.pass_conv(passable.view(batch * seq, 1, self.grid_size, self.grid_size)))
        pass_density = pass_density.view(batch, seq, self.grid_size * self.grid_size)

        feats = torch.cat([cell_emb, xy, is_food.unsqueeze(-1), pass_density.unsqueeze(-1)], dim=-1)
        feats = self.percell_mixer(feats)
        return feats, is_food, pass_density

    def _att_pool(
        self,
        feats: torch.Tensor,
        att_mlp: nn.Module,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = att_mlp(feats).squeeze(-1)
        if mask is not None:
            valid = mask.bool()
            scores = scores.masked_fill(~valid, -1e9)
            weights = F.softmax(scores, dim=-1)
            no_valid = (valid.sum(dim=-1, keepdim=True) == 0)
            if no_valid.any():
                weights = torch.where(no_valid, torch.zeros_like(weights), weights)
        else:
            weights = F.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * feats).sum(dim=-2)
        return pooled, weights

    def _food_pointer(self, w_food: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xy = self.cell_xy
        pointer = torch.matmul(w_food, xy)
        center = self.center_xy.view(1, 1, 2).expand_as(pointer)
        delta = pointer - center
        distance = torch.norm(delta, dim=-1, keepdim=True)
        return delta, distance

    def _per_step(self, grid_flat: torch.Tensor, state_seq: torch.Tensor) -> tuple[torch.Tensor, dict]:
        grid_ids = grid_flat.long().clamp(min=0, max=self.num_cell_types - 1)
        feats, is_food, pass_density = self._percell_features(grid_ids)

        pooled_all, _ = self._att_pool(feats, self.att_all)
        pooled_food, food_weights = self._att_pool(feats, self.att_food, mask=(is_food > 0))

        has_food = (is_food.sum(dim=-1, keepdim=True) > 0).float()
        food_weights = food_weights * has_food

        delta, distance = self._food_pointer(food_weights)

        grid_step = self.grid_step_proj(pooled_all)
        state_norm = self.state_norm(state_seq)
        grid_norm = self.grid_norm(grid_step)
        geo = torch.cat([delta, distance], dim=-1)
        fused_input = torch.cat([grid_norm, state_norm, geo], dim=-1)
        gate = torch.sigmoid(self.fuse_gate(fused_input))
        fused = torch.tanh(self.fuse_proj(fused_input))
        z = gate * fused + (1.0 - gate) * grid_norm

        helpers = {
            "grid_ids": grid_ids,
            "delta": delta,
            "distance": distance,
            "has_food": has_food,
        }
        return z, helpers

    def _selector(self, hidden: torch.Tensor, state_seq: torch.Tensor, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        if which == "actor":
            q = torch.tanh(self.sel_actor_h(hidden) + self.sel_actor_s(state_seq))
            raw = self.sel_actor_v(q).squeeze(-1)
        else:
            q = torch.tanh(self.sel_critic_h(hidden) + self.sel_critic_s(state_seq))
            raw = self.sel_critic_v(q).squeeze(-1)

        if self.selector_recency != 0.0:
            ramp = torch.linspace(0.0, 1.0, steps=hidden.size(1), device=hidden.device).view(1, -1)
            raw = raw + self.selector_recency * ramp

        tau = max(self.selector_tau, 1e-4)
        weights = F.softmax(raw / tau, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights

    def _dir_neighbor_ids(self, grid_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        g = grid_ids.view(-1, self.sequence_length, self.grid_size, self.grid_size)
        c = self.grid_size // 2
        up1, down1, left1, right1 = g[:, :, c - 1, c], g[:, :, c + 1, c], g[:, :, c, c - 1], g[:, :, c, c + 1]
        up2, down2, left2, right2 = g[:, :, c - 2, c], g[:, :, c + 2, c], g[:, :, c, c - 2], g[:, :, c, c + 2]
        return up1, down1, left1, right1, up2, down2, left2, right2

    def _immediate_affordance(self, grid_ids: torch.Tensor) -> torch.Tensor:
        g = grid_ids.view(-1, self.sequence_length, self.grid_size, self.grid_size)
        c = self.grid_size // 2
        up1, down1, left1, right1 = g[:, :, c - 1, c], g[:, :, c + 1, c], g[:, :, c, c - 1], g[:, :, c, c + 1]
        emb_up = self.affordance(up1)
        emb_down = self.affordance(down1)
        emb_left = self.affordance(left1)
        emb_right = self.affordance(right1)
        return torch.stack([
            emb_up[..., 0],
            emb_down[..., 1],
            emb_left[..., 2],
            emb_right[..., 3],
        ], dim=-1)

    def _food_direction_bias(self, delta_selected: torch.Tensor) -> torch.Tensor:
        dx, dy = delta_selected[:, 0], delta_selected[:, 1]
        return torch.stack([-dy, dy, -dx, dx], dim=-1)

    def _raycast_bias(self, grid_ids: torch.Tensor) -> torch.Tensor:
        g = grid_ids.view(-1, self.sequence_length, self.grid_size, self.grid_size)
        c = self.grid_size // 2
        steps = min(self.ray_steps, self.grid_size - 1)

        def scan(offsets: list[tuple[int, int]]) -> torch.Tensor:
            vals = []
            for dr, dc in offsets:
                r, col = c + dr, c + dc
                if 0 <= r < self.grid_size and 0 <= col < self.grid_size:
                    vals.append(g[:, :, r, col])
                else:
                    vals.append(torch.full_like(g[:, :, 0, 0], fill_value=-1))
            return torch.stack(vals, dim=0).permute(1, 2, 0)

        up_offsets = [(-k, 0) for k in range(1, steps + 1)]
        down_offsets = [(k, 0) for k in range(1, steps + 1)]
        left_offsets = [(0, -k) for k in range(1, steps + 1)]
        right_offsets = [(0, k) for k in range(1, steps + 1)]

        up_seq, down_seq, left_seq, right_seq = map(
            scan,
            (up_offsets, down_offsets, left_offsets, right_offsets),
        )

        def score_from_seq(seq: torch.Tensor) -> torch.Tensor:
            is_food = seq == self.food_id
            is_empty = seq == self.empty_id
            is_move = seq == self.moveable_id
            passable = is_empty | is_food | is_move

            pass_float = passable.float()
            corridor = torch.cumprod(pass_float + 1e-6, dim=-1)
            corridor_len = (corridor > 0.5).float().sum(dim=-1)

            any_food = is_food.any(dim=-1)
            seen = torch.cumsum(is_food.float(), dim=-1)
            first_food = (seen == 1.0) & is_food
            idxs = torch.arange(1, steps + 1, device=seq.device, dtype=torch.float32).view(1, 1, -1)
            dist_found = (first_food.float() * idxs).max(dim=-1).values
            dist = torch.where(any_food, torch.where(dist_found > 0, dist_found, dist_found.new_full(dist_found.shape, steps + 1)), steps + 1.0)

            dist_score = 1.0 / (dist + 1.0)
            corridor_score = corridor_len / float(steps)
            return self.ray_alpha * dist_score + self.ray_beta * corridor_score

        dir_score = torch.stack(
            [score_from_seq(up_seq), score_from_seq(down_seq), score_from_seq(left_seq), score_from_seq(right_seq)],
            dim=-1,
        )
        return dir_score.mean(dim=1)

    def _action_affordance(self, grid_ids: torch.Tensor, state_seq: torch.Tensor) -> torch.Tensor:
        up1, down1, left1, right1, up2, down2, left2, right2 = self._dir_neighbor_ids(grid_ids)
        is_empty = lambda x: x == self.empty_id
        is_food = lambda x: x == self.food_id
        is_move = lambda x: x == self.moveable_id
        is_grab = lambda x: x == self.grabbable_id

        def move_forward(n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
            return (is_empty(n1) | is_food(n1) | (is_move(n1) & is_empty(n2)))

        move_any = move_forward(up1, up2) | move_forward(down1, down2) | move_forward(left1, left2) | move_forward(right1, right2)
        grab_any = is_grab(up1) | is_grab(down1) | is_grab(left1) | is_grab(right1)
        place_any = is_empty(up1) | is_empty(down1) | is_empty(left1) | is_empty(right1)

        inventory_frac = state_seq[..., 2]
        can_grab = grab_any & (inventory_frac < 0.999)
        can_place = place_any & (inventory_frac > 0.001)

        move_bias = move_any.float()
        grab_bias = can_grab.float()
        place_bias = can_place.float()
        return torch.stack([move_bias, grab_bias, place_bias], dim=-1)

    def forward(self, grid_flat: torch.Tensor, state_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, hw = grid_flat.shape
        if hw != self.hw:
            raise ValueError(f"Expected flattened grid size {self.hw}, received {hw}")

        per_step, helpers = self._per_step(grid_flat, state_seq)
        grid_ids = helpers["grid_ids"]
        delta = helpers["delta"]
        has_food = helpers["has_food"]

        hidden, _ = self.gru(per_step)
        hidden = self.post_gru(hidden)
        hidden_last = hidden[:, -1, :]

        ctx_actor, weights_actor = self._selector(hidden, state_seq, which="actor")
        self._last_selector_weights = weights_actor.detach()
        actor_in = torch.cat([ctx_actor, hidden_last], dim=-1)

        action_logits = self.pi_action(actor_in)
        direction_logits = self.pi_direction(actor_in)

        delta_selected = torch.bmm(weights_actor.unsqueeze(1), delta).squeeze(1)
        dir_bias_food = self._food_direction_bias(delta_selected) * self.dir_bias_scale
        afford = self._immediate_affordance(grid_ids)
        dir_bias_aff = torch.bmm(weights_actor.unsqueeze(1), afford).squeeze(1) * self.affordance_scale
        dir_bias_ray = self._raycast_bias(grid_ids)

        prob_food_selected = torch.bmm(weights_actor.unsqueeze(1), has_food).squeeze(1).squeeze(-1)
        mask_food = (prob_food_selected > 0.5).float().unsqueeze(-1)
        dir_bias_food = dir_bias_food * mask_food

        gate_dir = torch.sigmoid(self.gate_dir(actor_in))
        direction_logits = direction_logits + gate_dir * (dir_bias_food + dir_bias_aff + dir_bias_ray)

        afford_action = self._action_affordance(grid_ids, state_seq)
        act_bias = torch.bmm(weights_actor.unsqueeze(1), afford_action).squeeze(1)
        gate_act = torch.sigmoid(self.gate_act(actor_in))
        action_logits = action_logits + gate_act * (self.action_bias_scale * act_bias)

        ctx_value, _ = self._selector(hidden, state_seq, which="critic")
        value_feat = self.critic_pre(ctx_value)
        value_raw = self.v_head_raw(value_feat).squeeze(-1)
        value = torch.tanh(value_raw) * self.v_max

        return action_logits, direction_logits, value

    def get_halting_aux(self, state: tuple[torch.Tensor, torch.Tensor]) -> dict:
        if self._last_selector_weights is None:
            return {}
        weights = self._last_selector_weights
        halt_prob = weights.max(dim=-1, keepdim=True).values
        segments_used = torch.ones(weights.size(0), dtype=torch.long, device=weights.device)
        return {
            "selector_weights": weights,
            "halt_prob": halt_prob,
            "segments_used": segments_used,
        }


YourModelClass = CoordPointerGRU  # Backwards compatible alias


def _resolve_model_class() -> Type[nn.Module]:
    class_name = getattr(Config, "USER_MODEL_CLASS", None)
    if not class_name:
        return CoordPointerGRU

    module_name = getattr(Config, "USER_MODEL_MODULE", "fun_grid.user_model")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Warning: user model module '{module_name}' not found. Falling back to default model.")
        return CoordPointerGRU

    try:
        model_cls = getattr(module, class_name)
    except AttributeError:
        print(
            f"Warning: user model class '{class_name}' not found in '{module_name}'. "
            "Falling back to default model."
        )
        return CoordPointerGRU

    if not issubclass(model_cls, nn.Module):
        print(f"Warning: user model '{class_name}' is not an nn.Module. Using default model.")
        return CoordPointerGRU
    return model_cls


class PPOAgent(nn.Module):
    """Wrapper that exposes PPO-friendly helpers around the core policy network."""

    def __init__(self) -> None:
        super().__init__()
        model_cls = _resolve_model_class()
        self.agent_model: nn.Module = model_cls().to(Config.DEVICE)

    def forward(self, grid_input: torch.Tensor, state_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, height, width = grid_input.shape
        grid_flat = grid_input.reshape(batch, seq, height * width)
        action_logits, direction_logits, value = self.agent_model(
            grid_flat.to(Config.DEVICE),
            state_input.to(Config.DEVICE),
        )
        action_probs = F.softmax(action_logits, dim=-1)
        direction_probs = F.softmax(direction_logits, dim=-1)
        return action_probs, direction_probs, value

    def get_new_log_probs(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        action_idx: torch.Tensor,
        direction_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_inputs, state_inputs = state
        batch, seq, height, width = grid_inputs.shape
        grid_flat = grid_inputs.reshape(batch, seq, height * width)

        action_logits, direction_logits, values = self.agent_model(
            grid_flat.to(Config.DEVICE),
            state_inputs.to(Config.DEVICE),
        )

        action_dist = Categorical(logits=action_logits)
        direction_dist = Categorical(logits=direction_logits)

        action_log_prob = action_dist.log_prob(action_idx.to(Config.DEVICE))
        direction_log_prob = direction_dist.log_prob(direction_idx.to(Config.DEVICE))
        entropy = action_dist.entropy().mean() + direction_dist.entropy().mean()

        for tensor in (action_log_prob, direction_log_prob, values):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                tensor.nan_to_num_(0.0)

        return action_log_prob, direction_log_prob, values, entropy

    def get_value(self, state: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        grid_inputs, state_inputs = state
        batch, seq, height, width = grid_inputs.shape
        grid_flat = grid_inputs.reshape(batch, seq, height * width)
        _, _, values = self.agent_model(
            grid_flat.to(Config.DEVICE),
            state_inputs.to(Config.DEVICE),
        )
        return values

    def get_halting_aux(self, state: tuple[torch.Tensor, torch.Tensor]) -> dict:
        if hasattr(self.agent_model, "get_halting_aux"):
            return self.agent_model.get_halting_aux(state)
        return {}

    def save(self, name_or_epoch: int | str | None = None) -> None:
        import os

        os.makedirs("models", exist_ok=True)
        if isinstance(name_or_epoch, (int, str)):
            path = f"models/ppo_model_{Config.MODEL_NAME}_{name_or_epoch}.pth"
        else:
            path = f"models/ppo_model_{Config.MODEL_NAME}.pth"
        torch.save(self.state_dict(), path)
        print(f"Saved model to {path}")

    def load(self, path: str) -> None:
        import os

        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=Config.DEVICE))
            print(f"Loaded model from {path}")
        else:
            print(f"No model found at {path}. Training new model.")

    def calculate_trainable_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


__all__ = ["CoordPointerGRU", "YourModelClass", "PPOAgent", "RMSNorm1d", "MLP"]
