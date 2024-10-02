# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from typing import Dict, Generator

import gymnasium as gym
import numpy as np
import torch as th
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from rllte.common.prototype import BaseStorage
from rllte.common.type_alias import TwoHeadEpisodicRolloutBatch


class TwoHeadEpisodicRolloutStorage(BaseStorage):
    """Episodic rollout storage for on-policy algorithms that use an LSTM.
    It is the same as VanillaRolloutStorage but samples enitre trajectories instead of batches of different steps.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device to convert the data.
        storage_size (int): The capacity of the storage. Here it refers to the length of per rollout.
        batch_size (int): Batch size of samples.
        num_envs (int): The number of parallel environments.
        discount (float): The discount factor.
        gae_lambda (float): Weighting coefficient for generalized advantage estimation (GAE).

    Returns:
        Vanilla rollout storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        storage_size: int = 256,
        batch_size: int = 64,
        num_envs: int = 8,
        int_discount: float = 0.99,
        discount: float = 0.999,
        gae_lambda: float = 0.95,
    ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)
        self.discount = discount
        self.int_discount = int_discount
        self.gae_lambda = gae_lambda

        self.reset()

    def reset(self) -> None:
        """Reset the storage."""
        # data containers
        self.observations = th.empty(
            size=(self.storage_size + 1, self.num_envs, *self.obs_shape), dtype=th.float32, device=self.device
        )
        self.actions = th.empty(size=(self.storage_size, self.num_envs, self.action_dim), dtype=th.float32, device=self.device)
        self.rewards = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.terminateds = th.empty(size=(self.storage_size + 1, self.num_envs), dtype=th.float32, device=self.device)
        self.truncateds = th.empty(size=(self.storage_size + 1, self.num_envs), dtype=th.float32, device=self.device)
        # first next_terminated
        self.terminateds[0].fill_(0.0)
        self.truncateds[0].fill_(0.0)
        # extra part
        self.log_probs = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.values = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.returns = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.advantages = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        # intrinsic part 
        self.intrinsic_rewards = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.intrinsic_values = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.intrinsic_returns = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        self.intrinsic_advantages = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        # this will be filled by adding intrinsic advantages and extrinsic advantages
        self.combined_advantages = th.empty(size=(self.storage_size, self.num_envs), dtype=th.float32, device=self.device)
        super().reset()

    def add(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        infos: Dict,
        next_observations: th.Tensor,
        log_probs: th.Tensor,
        values: th.Tensor,
        intrinsic_values: th.Tensor,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            observations (th.Tensor): Observations.
            actions (th.Tensor): Actions.
            rewards (th.Tensor): Rewards.
            terminateds (th.Tensor): Termination signals.
            truncateds (th.Tensor): Truncation signals.
            infos (Dict): Extra information.
            next_observations (th.Tensor): Next observations.
            log_probs (th.Tensor): Log of the probability evaluated at `actions`.
            values (th.Tensor): Estimated values.

        Returns:
            None.
        """
        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions.view(self.num_envs, self.action_dim))
        self.rewards[self.step].copy_(rewards)
        self.terminateds[self.step + 1].copy_(terminateds)
        self.truncateds[self.step + 1].copy_(truncateds)
        self.observations[self.step + 1].copy_(next_observations)
        self.log_probs[self.step].copy_(log_probs)
        self.values[self.step].copy_(values.flatten())
        self.intrinsic_values[self.step].copy_(intrinsic_values.flatten())

        self.full = True if self.step == self.storage_size - 1 else False
        self.step = (self.step + 1) % self.storage_size

    def update(self) -> None:
        """Update the terminal state of each env."""
        self.terminateds[0].copy_(self.terminateds[-1])
        self.truncateds[0].copy_(self.truncateds[-1])

    def compute_intrinsic_returns_and_advantages(self, last_values: th.Tensor, episodic=True) -> None:
        """Perform generalized advantage estimation (GAE).

        Args:
            last_values (th.Tensor): Estimated values of the last step.

        Returns:
            None.
        """
        gae = 0
        for step in reversed(range(self.storage_size)):
            if step == self.storage_size - 1:
                next_values = last_values[:, 0]
            else:
                next_values = self.intrinsic_values[step + 1]

            # cut off the advantage calculation at the end of the episode if episodic
            if episodic:
                next_non_terminal = 1.0 - self.terminateds[step + 1]                
            else:
                next_non_terminal = 1.0
            
            delta = self.intrinsic_rewards[step] + self.int_discount * next_values * next_non_terminal - self.intrinsic_values[step]
            gae = delta + self.int_discount * self.gae_lambda * next_non_terminal * gae
            # time limit
            gae = gae * (1.0 - self.truncateds[step + 1])
            self.intrinsic_advantages[step] = gae
        self.intrinsic_returns = self.intrinsic_advantages + self.intrinsic_values

    def compute_extrinsic_returns_and_advantages(self, last_values: th.Tensor) -> None:
        """Perform generalized advantage estimation (GAE).

        Args:
            last_values (th.Tensor): Estimated values of the last step.

        Returns:
            None.
        """
        gae = 0
        for step in reversed(range(self.storage_size)):
            if step == self.storage_size - 1:
                next_values = last_values[:, 0]
            else:
                next_values = self.values[step + 1]
                
            # extrinsic rewards are always episodic
            next_non_terminal = 1.0 - self.terminateds[step + 1]
            
            delta = self.rewards[step] + self.discount * next_values * next_non_terminal - self.values[step]
            gae = delta + self.discount * self.gae_lambda * next_non_terminal * gae
            # time limit
            gae = gae * (1.0 - self.truncateds[step + 1])
            self.advantages[step] = gae

        self.returns = self.advantages + self.values

    def sample(self) -> Generator:
        """
        Choose a minibatch of environment indices and sample the entire rollout for those minibatches.
        By not sampling uniform transitions, we can now train an LSTM model on entire trajectories
        """

        assert self.full, "Cannot sample when the storage is not full!"
        # only sample random envs (entire rollout)
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_envs)), self.num_envs // 4, drop_last=True)

        b_obs = self.observations[:-1].reshape(-1, *self.obs_shape)
        b_act = self.actions.reshape(-1, *self.action_shape)
        b_ret = self.returns.reshape(-1)
        b_int_ret = self.intrinsic_returns.reshape(-1)
        b_val = self.values.reshape(-1)
        b_int_val = self.intrinsic_values.reshape(-1)
        b_ter = self.terminateds[:-1].reshape(-1)
        b_tru = self.truncateds[:-1].reshape(-1)
        b_log = self.log_probs.reshape(-1)
        b_adv = self.combined_advantages.view(-1)

        flat_idcs = np.arange(self.num_envs * self.storage_size).reshape(self.storage_size, self.num_envs)

        for indices in sampler:
            mb_inds = flat_idcs[:, indices].ravel()
            
            # normalize advantages in the minibatch
            adv_targ = (b_adv[mb_inds] - b_adv[mb_inds].mean()) / (b_adv[mb_inds].std() + 1e-8)

            yield TwoHeadEpisodicRolloutBatch(
                observations=b_obs[mb_inds],
                actions=b_act[mb_inds],
                values=b_val[mb_inds],
                returns=b_ret[mb_inds],
                intrinsic_values=b_int_val[mb_inds],
                intrinsic_returns=b_int_ret[mb_inds],
                terminateds=b_ter[mb_inds],
                truncateds=b_tru[mb_inds],
                old_log_probs=b_log[mb_inds],
                adv_targ=adv_targ,
                env_inds=indices,
            )