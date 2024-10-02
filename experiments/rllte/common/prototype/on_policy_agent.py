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


from collections import deque
from typing import Any, Deque, Dict, List, Optional
from copy import deepcopy

import numpy as np
import torch as th
import imageio

from rllte.common import utils
from rllte.common.prototype.base_agent import BaseAgent
from rllte.common.type_alias import OnPolicyType, RolloutStorageType, VecEnv

class OnPolicyAgent(BaseAgent):
    """Trainer for on-policy algorithms.

    Args:
        env (VecEnv): Vectorized environments for training.
        eval_env (VecEnv): Vectorized environments for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on pre-training model or not.
        num_steps (int): The sample length of per rollout.

    Returns:
        On-policy agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_steps: int = 128,
        use_lstm: bool = False,
    ) -> None:
        super().__init__(env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=pretraining)
        self.num_steps = num_steps
        # attr annotations
        self.policy: OnPolicyType
        self.storage: RolloutStorageType
        self.use_lstm = use_lstm

    def update(self) -> None:
        """Update the agent. Implemented by individual algorithms."""
        raise NotImplementedError

    def train(
        self,
        num_train_steps: int,
        init_model_path: Optional[str] = None,
        log_interval: int = 1,
        eval_interval: int = 100,
        save_interval: int = 100,
        num_eval_episodes: int = 10,
        th_compile: bool = True,
        anneal_lr: bool = True
    ) -> None:
        """Training function.

        Args:
            num_train_steps (int): The number of training steps.
            init_model_path (Optional[str]): The path of the initial model.
            log_interval (int): The interval of logging.
            eval_interval (int): The interval of evaluation.
            save_interval (int): The interval of saving model.
            num_eval_episodes (int): The number of evaluation episodes.
            th_compile (bool): Whether to use `th.compile` or not.
            anneal_lr (bool): Whether to anneal the learning rate or not.

        Returns:
            None.
        """
        # freeze the agent and get ready for training
        self.freeze(init_model_path=init_model_path, th_compile=th_compile)

        # reset the env        
        episode_rewards: Deque = deque(maxlen=10)
        intrinsic_episode_rewards: Deque = deque(maxlen=10)
        episode_steps: Deque = deque(maxlen=10)
        episode_achievements: Dict[str, deque] = {}
        obs, infos = self.env.reset(seed=self.seed)
        # get number of updates
        num_updates = int(num_train_steps // self.num_envs // self.num_steps)

        ###############################################################################################################
        # init obs normalization parameters if necessary
        if self.irs is not None:
            self.env = self.irs.init_normalization(self.num_steps, 20, self.env, obs)
        ###############################################################################################################
                    
        # only if using lstm, initialize lstm state
        if self.use_lstm:
            lstm_state = (
                th.zeros(self.policy.lstm.num_layers, self.num_envs, self.policy.lstm.hidden_size).to(self.device),
                th.zeros(self.policy.lstm.num_layers, self.num_envs, self.policy.lstm.hidden_size).to(self.device),
            )
            done = th.zeros(self.num_envs, dtype=th.bool, device=self.device)

        for update in range(num_updates):
            # important for updating the policy lstm later
            if self.use_lstm:
                self.initial_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())

            # try to eval
            if (update % eval_interval) == 0 and (self.eval_env is not None):
                eval_metrics = self.eval(num_eval_episodes)
                # log to console
                self.logger.eval(msg=eval_metrics)
            
            # update the learning rate
            if anneal_lr:
                for key in self.policy.optimizers.keys():
                    utils.linear_lr_scheduler(self.policy.optimizers[key], update, num_updates, self.lr)

            for _ in range(self.num_steps):
                # sample actions
                with th.no_grad(), utils.eval_mode(self):
                    if self.use_lstm:
                        actions, extra_policy_outputs = self.policy(obs, lstm_state, done, training=True)
                        lstm_state = extra_policy_outputs["lstm_state"]
                        del extra_policy_outputs["lstm_state"]
                    else:
                        actions, extra_policy_outputs = self.policy(obs, training=True)
                    
                    # observe rewards and next obs
                    next_obs, rews, terms, truncs, infos = self.env.step(actions)

                    if self.use_lstm:
                        done = th.logical_or(terms, truncs)

                # pre-training mode
                if self.pretraining:
                    rews = th.zeros_like(rews, device=self.device)

###############################################################################################################
                # adapt to intrinsic reward modules
                if self.irs is not None:
                    self.irs.watch(obs, actions, rews, terms, truncs, next_obs)
###############################################################################################################

                # add transitions
                self.storage.add(obs, actions, rews, terms, truncs, infos, next_obs, **extra_policy_outputs)

                # get episode information
                eps_r, eps_l = utils.get_episode_statistics(infos)
                episode_rewards.extend(eps_r)
                episode_steps.extend(eps_l)

###############################################################################################################
                # only used for craftax
                eps_achievements = utils.get_achievement_statistics(infos)
                for key, value in eps_achievements.items():
                    if key not in episode_achievements:
                        episode_achievements[key] = deque(maxlen=10)
                    episode_achievements[key].extend(value)
###############################################################################################################

                # set the current observation
                obs = next_obs

            # get the value estimation of the last step
            with th.no_grad():
                if self.use_lstm:
                    last_values = self.policy.get_value(next_obs, lstm_state, done).detach()
                else:
                    last_values = self.policy.get_value(next_obs).detach()

###############################################################################################################
            if self.irs is not None:
                # deal with the intrinsic reward module
                intrinsic_rewards = self.irs.compute(
                    samples={
                        "observations": self.storage.observations[:-1],  # type: ignore
                        "actions": self.storage.actions,
                        "rewards": self.storage.rewards,
                        "terminateds": self.storage.terminateds,
                        "truncateds": self.storage.truncateds,
                        "next_observations": self.storage.observations[1:],  # type: ignore
                    }
                )
                # just plus the intrinsic rewards to the extrinsic rewards
                self.storage.rewards += intrinsic_rewards.to(self.device)
                intrinsic_episode_rewards.extend([np.mean(intrinsic_rewards.cpu().numpy())])
###############################################################################################################
            
            # compute advantages and returns 
            if self.irs and self.pretraining and self.irs.rff is not None:
                self.storage.compute_returns_and_advantages(last_values, episodic=False)
            else:
                self.storage.compute_returns_and_advantages(last_values)

            # update the agent
            self.update()

            # update the storage
            self.storage.update()

            # log training information
            self.global_episode += self.num_envs
            self.global_step += self.num_envs * self.num_steps

            if len(episode_rewards) > 0 and update % log_interval == 0:
                total_time = self.timer.total_time()

                # log to console
                train_metrics = {
                    "step": self.global_step,
                    "episode": self.global_episode,
                    "episode_length": np.mean(list(episode_steps)),
                    "intrinsic_episode_reward": np.mean(list(intrinsic_episode_rewards)),
                    "episode_reward": np.mean(list(episode_rewards)),
                    "fps": self.global_step / total_time,
                    "total_time": total_time,
                }
###############################################################################################################
                if len(episode_achievements) > 0:
                    # only used for craftax
                    self.logger.additional(
                        msg={key: np.mean(value) for key, value in episode_achievements.items() if len(value) > 0}
                    )
###############################################################################################################
                self.logger.train(msg=train_metrics)
                self.logger.loss(msg=self.logger.metrics)

            # save model
            if update % save_interval == 0:
                self.save()

        # final save
        self.save()
        self.logger.info("Training Accomplished!")
        self.logger.info(f"Model saved at: {self.work_dir / 'model'}")

        # close env
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()

    def eval(self, num_eval_episodes: int) -> Dict[str, Any]:
        """Evaluation function.

        Args:
            num_eval_episodes (int): The number of evaluation episodes.

        Returns:
            The evaluation results.
        """
        assert self.eval_env is not None, "No evaluation environment is provided!"

        # reset the env
        obs, infos = self.eval_env.reset(seed=self.seed)
        episode_rewards: List[float] = []
        episode_steps: List[int] = []
        images: List[np.ndarray] = []
        
        # since render() is not implemented, use image observation
        img = obs.cpu().numpy()[0].transpose(1, 2, 0)

        if self.use_lstm:
            lstm_state = (
                th.zeros(self.policy.lstm.num_layers, self.num_envs, self.policy.lstm.hidden_size).to(self.device),
                th.zeros(self.policy.lstm.num_layers, self.num_envs, self.policy.lstm.hidden_size).to(self.device),
            )
            done = th.zeros(self.num_envs, dtype=th.bool, device=self.device)

        # evaluation loop
        while len(episode_rewards) < num_eval_episodes:
            images.append(img)
            
            with th.no_grad(), utils.eval_mode(self):
                if self.use_lstm:
                    actions, extra_policy_outputs = self.policy(obs, lstm_state, done, training=True)
                    lstm_state = extra_policy_outputs["lstm_state"]
                    del extra_policy_outputs["lstm_state"]
                else:
                    actions, _ = self.policy(obs, training=True)

                
                next_obs, rews, terms, truncs, infos = self.eval_env.step(actions)
                
                if self.use_lstm:
                    done = th.logical_or(terms, truncs)
                
                # since render() is not implemented, use image observation
                img = next_obs.cpu().numpy()[0].transpose(1, 2, 0)

            # get episode information
            if "episode" in infos:
                eps_r, eps_l = utils.get_episode_statistics(infos)
                episode_rewards.extend(eps_r)
                episode_steps.extend(eps_l)

                if self.use_lstm:
                    lstm_state = (
                        th.zeros(self.policy.lstm.num_layers, self.num_envs, self.policy.lstm.hidden_size).to(self.device),
                        th.zeros(self.policy.lstm.num_layers, self.num_envs, self.policy.lstm.hidden_size).to(self.device),
                    )
                    done = th.zeros(self.num_envs, dtype=th.bool, device=self.device)

            # set the current observation
            obs = next_obs

        # save gif
        imageio.mimsave(f"{self.work_dir}/eval_{self.global_step}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=10)

        return {
            "step": self.global_step,
            "episode": self.global_episode,
            "episode_length": np.mean(episode_steps),
            "episode_reward": np.mean(episode_rewards),
            "total_time": self.timer.total_time(),
        }
