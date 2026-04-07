# MiniGrid environment wrapper for GamingAgent
# MiniGrid provides a family of gridworld environments for reinforcement learning research

from typing import Any, Dict, Tuple, Optional, List
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

# Import the adapter and Observation dataclass
from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation


class MiniGridEnv(gym.Env):
    """
    Wrapper for MiniGrid environments that integrates with GamingAgent's adapter system.

    MiniGrid provides partially observable gridworld environments where the agent
    must navigate, pick up objects, and complete tasks. Popular environments include:
    - MiniGrid-Empty-*: Navigate to goal in empty room
    - MiniGrid-DoorKey-*: Find key, unlock door, reach goal
    - MiniGrid-MultiRoom-*: Navigate through multiple rooms
    - MiniGrid-FourRooms: Classic four rooms environment
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        env_name: str = "MiniGrid-Empty-8x8-v0",
        max_steps: int | None = None,
        # Adapter related parameters (will be passed by the runner)
        game_name_for_adapter: str = "minigrid",
        observation_mode_for_adapter: str = "vision",
        agent_cache_dir_for_adapter: str = "cache/minigrid/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_07_minigrid/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
    ) -> None:
        """
        Initialize MiniGrid environment wrapper.

        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            env_name: Name of the MiniGrid environment to create
            max_steps: Maximum steps per episode (uses environment default if None)
            game_name_for_adapter: Game identifier for the adapter
            observation_mode_for_adapter: Observation mode ("vision", "text", or "both")
            agent_cache_dir_for_adapter: Directory for caching agent observations
            game_specific_config_path_for_adapter: Path to game-specific config
            max_stuck_steps_for_adapter: Maximum steps without progress before termination
        """
        # Create the base MiniGrid environment
        self.env_name = env_name
        self.base_env = gym.make(env_name, render_mode=render_mode, max_steps=max_steps)

        # Wrap to get RGB images for visual observations
        # RGBImgObsWrapper converts the partial observation to full RGB image
        self.base_env = RGBImgObsWrapper(self.base_env)

        self.render_mode = render_mode

        # Set up spaces from wrapped environment
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        # Store environment state
        self.current_obs = None
        self.current_info_dict: Dict[str, Any] = {}

        # Initialize the adapter
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """
        Reset the environment and return initial observation via adapter.

        Returns:
            Observation object and info dict from adapter
        """
        # Reset the base environment
        obs, info = self.base_env.reset(seed=seed, options=options)

        # Store current state
        self.current_obs = obs
        self.current_info_dict = info

        # Use adapter to process and return observation
        agent_obs, agent_info = self.adapter.reset(
            gym_obs=obs,
            gym_info=info
        )

        return agent_obs, agent_info

    def step(
        self, action: str
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute action in environment and return result via adapter.

        Args:
            action: Action string from agent (e.g., "turn_left", "forward", "pickup")

        Returns:
            Observation, reward, terminated, truncated, info from adapter
        """
        # Adapter converts action string to environment action index
        agent_obs, reward, terminated, truncated, info = self.adapter.step(
            action_str=action,
            base_env=self.base_env
        )

        # Store the latest gym observation and info from the step
        if hasattr(self.adapter, 'current_gym_obs'):
            self.current_obs = self.adapter.current_gym_obs
        if hasattr(self.adapter, 'current_gym_info'):
            self.current_info_dict = self.adapter.current_gym_info

        return agent_obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the environment."""
        return self.base_env.render()

    def close(self):
        """Close the environment."""
        if self.base_env:
            self.base_env.close()

    def get_textual_representation(self) -> str:
        """
        Generate textual representation of current state for text-based observations.

        Returns:
            String describing the current state
        """
        if self.current_obs is None:
            return "Environment not initialized. Please reset first."

        # Try to get mission/task description
        mission = self.current_info_dict.get('mission', 'No mission specified')

        # Get agent's direction
        direction_map = {0: "right", 1: "down", 2: "left", 3: "up"}
        agent_dir = direction_map.get(
            self.base_env.unwrapped.agent_dir,
            "unknown"
        )

        # Get agent position
        agent_pos = self.base_env.unwrapped.agent_pos

        # Get carrying status
        carrying = self.base_env.unwrapped.carrying
        carrying_str = f"Carrying: {carrying.type if carrying else 'nothing'}"

        text = f"""MiniGrid Environment: {self.env_name}
Mission: {mission}
Agent Position: {agent_pos}
Agent Direction: {agent_dir}
{carrying_str}
Step Count: {self.base_env.unwrapped.step_count}

Available Actions:
- turn_left: Turn left 90 degrees
- turn_right: Turn right 90 degrees
- forward: Move forward one cell
- pickup: Pick up an object
- drop: Drop the object you're carrying
- toggle: Toggle/activate an object (e.g., open door)
- done: Declare task complete (use when you've reached the goal)

Note: You can only see what's in front of you. Explore by moving and turning.
"""
        return text

    def get_symbolic_representation(self) -> str:
        """
        Generate symbolic/ASCII representation of the environment state.

        Returns:
            ASCII art representation of the grid
        """
        if self.current_obs is None:
            return "Environment not initialized."

        try:
            # Get the full grid from the unwrapped environment
            grid = self.base_env.unwrapped.grid
            width = grid.width
            height = grid.height

            # Build ASCII representation
            lines = []
            lines.append("+" + "-" * width + "+")

            for j in range(height):
                line = "|"
                for i in range(width):
                    cell = grid.get(i, j)

                    # Check if agent is at this position
                    if (i, j) == tuple(self.base_env.unwrapped.agent_pos):
                        # Agent direction indicators
                        dir_chars = {0: ">", 1: "v", 2: "<", 3: "^"}
                        line += dir_chars.get(self.base_env.unwrapped.agent_dir, "@")
                    elif cell is None:
                        line += " "
                    elif cell.type == "wall":
                        line += "#"
                    elif cell.type == "door":
                        line += "D" if cell.is_open else "d"
                    elif cell.type == "key":
                        line += "k"
                    elif cell.type == "ball":
                        line += "o"
                    elif cell.type == "box":
                        line += "b"
                    elif cell.type == "goal":
                        line += "G"
                    elif cell.type == "lava":
                        line += "~"
                    else:
                        line += "?"
                line += "|"
                lines.append(line)

            lines.append("+" + "-" * width + "+")

            # Add legend
            lines.append("\nLegend:")
            lines.append("  @ = Agent (>, v, <, ^ for direction)")
            lines.append("  # = Wall")
            lines.append("  G = Goal")
            lines.append("  d/D = Door (closed/open)")
            lines.append("  k = Key")
            lines.append("  ~ = Lava")

            return "\n".join(lines)
        except Exception as e:
            return f"Error generating symbolic representation: {str(e)}"


# Register wrapper function for easy instantiation
def make_minigrid_env(
    env_name: str = "MiniGrid-Empty-8x8-v0",
    render_mode: str | None = None,
    **kwargs
) -> MiniGridEnv:
    """
    Factory function to create MiniGrid environment wrapper.

    Args:
        env_name: MiniGrid environment name
        render_mode: Rendering mode
        **kwargs: Additional arguments passed to MiniGridEnv

    Returns:
        Configured MiniGridEnv instance
    """
    return MiniGridEnv(
        env_name=env_name,
        render_mode=render_mode,
        **kwargs
    )
