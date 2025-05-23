from abc import abstractmethod
from typing import Any, Dict

import gym
import numpy as np

from dreamerv3.embodied.core.base import Env

"""
Abstract class for a SimDreamer simulation environment that can be reset to arbitrary internal states.
"""


class SimulationEnv(Env):

    def __len__(self):
        return 0  # Return positive integer for batched envs.

    def __bool__(self):
        return True  # Env is always truthy, despite length zero.

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'len={len(self)}, '
            f'obs_space={self.obs_space}, '
            f'act_space={self.act_space})')

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, gym.Space]:
        # The observation space must contain the keys is_first, is_last, and
        # is_terminal. Commonly, it also contains the keys reward and image. By
        # convention, keys starting with log_ are not consumed by the agent.

        # For SimulationEnv, this must also contain internal_state_space.
        raise NotImplementedError('Returns: dict of spaces')

    @property
    @abstractmethod
    def action_space(self) -> Dict[str, gym.Space]:
        # The observation space must contain the keys action and reset. This
        # restriction may be lifted in the future.
        raise NotImplementedError('Returns: dict of spaces')

    @property
    @abstractmethod
    def internal_state_space(self) -> gym.Space:
        """
        Defines the legal range of values for the SimulationEnv's internal state.
        This space should be tractable for a neural network to predict values for.
        All values within the defined space should be mappable to an actual sim state that a SimulationEnv can reset to.
        """
        return NotImplemented

    @abstractmethod
    def step(self, action):
        raise NotImplementedError('Returns: dict')

    @abstractmethod
    def get_internal_state(self) -> np.ndarray:
        """
        Returns:
            The current simulation internal state
        """
        raise NotImplemented

    @abstractmethod
    def reset_to_internal_state(self, new_internal_state: np.ndarray) -> (dict, bool, bool):
        """
        Resets the environment to a specific provided internal state.
        No details of the environment state should be carried over from conditions prior to this method call.
        If the internal_state_space is not exhaustive, environment details not specified in internal_state_space
        should be reset to a default value.

        Args:
            new_internal_state: The internal state to reset to.

        Returns:
            1. The current observation after resetting to the provided new_internal_state,
            or all zeros for each component if the sim can't reset to this provided state
            (must return False for both is_valid and is_usable in the latter case)

            2. is_valid - Whether the provided new_internal_state is a valid state for the simulation.
            It is ok to designate a state as invalid and still be able to reset to it/provide an observation for it.

            3. is_usable - Whether the provided new_internal_state can be reset to and used at all.
            If is_usable is False, is_valid must be False too.
        """
        raise NotImplemented

    @abstractmethod
    def validate_internal_state_input(self, internal_state_input: np.ndarray, normalize_if_enabled: True) -> np.ndarray:
        """
        If internal_state_input would result in invalid or unstable dynamics, return a corrected internal_state that
        is similar to the invalid input.

        If internal_state_input is already stable and valid, return it without modification.

        Args:
            internal_state_input: The internal state to correct if it is invalid or unstable:
            normalize_if_enabled: In environments where the internal_state_space is a normalized version of
                some other original internal state space, this controls whether to also normalize
                the output of this method.
                Useful to toggle off for internal usage within the environment implementation.

        Returns:
            A valid/stable internal_state that is similar to internal_state_input
        """
        raise NotImplemented

    def render(self, *args, **kwargs) -> Any:
        """
        Standard Gym render method. Supported functionality may vary by environment.
        """
        raise NotImplemented

    def close(self):
        pass
