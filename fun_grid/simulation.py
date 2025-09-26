"""High-level environment orchestration for training FunGrid agents."""

from __future__ import annotations

import gc
from typing import Iterable, List

import torch
from torch import optim

from .config import Config
from .entities import Agent
from .grid import GridEnvironment
from .rl import PPO
from .visualizer import GridVisualizer


class EnvironmentSimulation:
    """Manage multiple environments, agents, and PPO updates."""

    def __init__(self) -> None:
        self.envs = self.initialize_environments()
        self.visualizer = self.initialize_visualizer(self.envs)
        self.model = self.initialize_model()
        self.ppo = PPO(
            self.model,
            self.optimizer,
            Config.CLIP_PARAM,
            Config.PPO_EPOCHS,
            Config.TARGET_KL,
            Config.GAMMA,
            Config.TAU,
        )
        self.agents = self.initialize_agents(self.model, self.envs, self.visualizer)
        self._quit = False

    def initialize_environments(self) -> List[GridEnvironment]:
        return [GridEnvironment(idx) for idx in range(Config.NUM_ENVS)]

    def initialize_visualizer(self, envs) -> GridVisualizer | None:
        if Config.VISUALIZE:
            return GridVisualizer(envs)
        return None

    def initialize_model(self):
        model = PPOAgent()  # type: ignore[name-defined]
        if Config.LOAD_MODEL:
            model.load(Config.MODEL_PATH)
        calculated_trainable_params = model.calculate_trainable_parameters()
        print(f"Calculated trainable parameters: {calculated_trainable_params}")
        self.optimizer = self.initialize_optimizer(model)
        return model

    def initialize_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    def initialize_agents(self, model, envs, visualizer=None, initial_energy=None):
        agents = []
        for env_idx, env in enumerate(envs):
            for agent_idx in range(Config.NUM_AGENTS):
                agent = Agent(model, env, env_idx, agent_idx, visualizer, initial_energy or Agent.INITIAL_ENERGY)
                agents.append(agent)
        return agents

    def reset_agents(self, agents: Iterable) -> None:
        for agent in agents:
            agent.reset()
        for env in self.envs:
            env.new_grid_generated = False

    def run(self) -> None:
        episode = 1
        try:
            while not self._quit:
                if Config.VISUALIZE and self.visualizer is not None:
                    if not self.visualizer.process_events():
                        self._quit = True
                        break

                self.print_episode_header(episode)
                self.run_episode()
                if self._quit:
                    break

                self.print_episode_summary()
                if self.should_save_model(episode):
                    self.model.save(episode)
                episode += 1

        except KeyboardInterrupt:
            print("\n[INFO] Caught KeyboardInterrupt. Shutting down...")
        finally:
            if Config.VISUALIZE and self.visualizer is not None:
                self.visualizer.close()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def run_episode(self) -> None:
        for env in self.envs:
            env.reset()
        if Config.VISUALIZE and self.visualizer is not None:
            self.visualizer.reset()
        self.reset_agents(self.agents)
        gc.collect()

        ep_steps = 1
        while (not self._quit) and (not all(agent.done for agent in self.agents)):
            if Config.VISUALIZE and self.visualizer is not None:
                if not self.visualizer.process_events():
                    self._quit = True
                    break

            self.update_agents_and_envs(ep_steps)
            ep_steps += 1

    def update_agents_and_envs(self, steps: int) -> None:
        active_agents = [agent for agent in self.agents if not agent.done]
        if not active_agents:
            return

        view_tensors = torch.stack(
            [agent.prepare_view_tensor() for agent in active_agents], dim=0
        ).to(Config.DEVICE)
        state_vectors = torch.stack(
            [agent.prepare_state_vector_sequence() for agent in active_agents], dim=0
        ).to(Config.DEVICE)

        self.model.eval()
        with torch.no_grad():
            actions, directions, value_outputs = self.model(view_tensors, state_vectors)

        for idx, agent in enumerate(active_agents):
            state, action, direction, a_logp, d_logp, value, reward, done = agent.update(
                actions[idx], directions[idx], view_tensors[idx], state_vectors[idx], value_outputs[idx]
            )
            self.update_agent(agent, state, action, direction, a_logp, d_logp, reward, value, done)

        del view_tensors, state_vectors
        torch.cuda.empty_cache()

        need_update = (all(agent.done for agent in self.agents) or steps % Config.UPDATE_FREQUENCY == 0)
        if need_update and Config.USE_PPO_UPDATE:
            self.model.eval()
            with torch.no_grad():
                for agent in self.agents:
                    if agent.transitions.transitions and (not agent.done):
                        next_view_seq = (
                            torch.stack(list(agent.view_sequence), dim=0)
                            .unsqueeze(0)
                            .to(Config.DEVICE)
                        )
                        next_state_seq = (
                            torch.stack(list(agent.state_vector_sequence), dim=0)
                            .unsqueeze(0)
                            .to(Config.DEVICE)
                        )
                        next_value = self.model.get_value((next_view_seq, next_state_seq)).squeeze(0)
                        agent.transitions.mark_last_as_truncated(next_value)

            self.model.train()
            transitions = []
            for agent in self.agents:
                transitions.extend(agent.transitions.transitions)
                agent.transitions.clear_memory()

            if transitions:
                losses = self.ppo.update(transitions)
                for agent in self.agents:
                    agent.action_loss.append(losses.action_loss)
                    agent.direction_loss.append(losses.direction_loss)
                    agent.value_loss.append(losses.value_loss)
                    agent.compute_times.append(losses.compute_time)

        for env in self.envs:
            self.update_environment_for_agents(env, steps)

    def update_agent(self, agent, state, action, direction, action_log_prob, direction_log_prob, reward, value, done):
        if state is not None and Config.USE_PPO_UPDATE:
            agent.transitions.store_transition(
                state, action, direction, action_log_prob, direction_log_prob, reward, value, done
            )
        return

    def update_environment_for_agents(self, env, steps: int) -> None:
        agents = [agent for agent in self.agents if agent.env_idx == env.idx]
        if not all(agent.done for agent in agents):
            if Config.FOOD_TICK:
                env.update_food(steps)
            if len(env.bouncing_obstacle_objects) > 0:
                non_empty = env.build_non_empty_cells()
                env.update_bouncing_obstacles(steps, non_empty)
            if self.visualizer is not None:
                self.visualizer.update(agents)

    def print_episode_header(self, episode: int) -> None:
        print("-" * 20)
        print(f"Episode {episode}:")
        print("-" * 20)

    def print_episode_summary(self) -> None:
        print("Episode Summary:")
        if Config.PPO_TOGETHER:
            total_action_loss = sum(sum(agent.action_loss) for agent in self.agents)
            total_direction_loss = sum(sum(agent.direction_loss) for agent in self.agents)
            total_value_loss = sum(sum(agent.value_loss) for agent in self.agents)
            num_updates = sum(len(agent.action_loss) for agent in self.agents)

            mean_action_loss = total_action_loss / num_updates if num_updates > 0 else 0
            mean_direction_loss = total_direction_loss / num_updates if num_updates > 0 else 0
            mean_value_loss = total_value_loss / num_updates if num_updates > 0 else 0

            print(
                f"Global Mean Losses: Action={mean_action_loss:.4f}, "
                f"Direction={mean_direction_loss:.4f}, Value={mean_value_loss:.4f}"
            )

            for idx, agent in enumerate(self.agents):
                total_compute_time = sum(agent.compute_times)
                print(
                    f"  Agent {idx}: Reward={agent.total_reward:.3f}, Food={agent.food_eaten}, "
                    f"Steps={agent.steps}, Compute Time={total_compute_time:.2f}"
                )
        else:
            for idx, agent in enumerate(self.agents):
                mean_action_loss = sum(agent.action_loss) / len(agent.action_loss) if agent.action_loss else 0
                mean_direction_loss = sum(agent.direction_loss) / len(agent.direction_loss) if agent.direction_loss else 0
                mean_value_loss = sum(agent.value_loss) / len(agent.value_loss) if agent.value_loss else 0
                total_compute_time = sum(agent.compute_times)
                print(
                    f"Agent {idx}: Reward={agent.total_reward:.3f}, Food={agent.food_eaten}, "
                    f"Steps={agent.steps}, Action Loss={mean_action_loss:.4f}, "
                    f"Direction Loss={mean_direction_loss:.4f}, Value Loss={mean_value_loss:.4f}, "
                    f"Compute Time={total_compute_time:.2f}"
                )

    def should_save_model(self, episode: int) -> bool:
        return episode % Config.SAVE_FREQUENCY == 0 or episode == 1


__all__ = ["EnvironmentSimulation"]
