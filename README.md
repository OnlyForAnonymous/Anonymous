# File List

1. `api`: Packaged APIs.
    - `README.md`: Documentation;
    - `rllte/xplore/reward`: Implementations of the intrinsic rewards;
    - `LICENSE`: Project license;
    - `pyproject.toml`: Configuration file;
    - `0 quick_start.ipynb`: Quick start;
    - `1 rlexplore_with_rllte.ipynb`: RLeXplore with RLLTE;
    - `2 rlexplore_with_sb3.ipynb`: RLeXplore with Stable-Baselines3;
    - `3 rlexplore_with_cleanrl.py`: RLeXplore with CleanRL;
    - `4 mixed_intrinsic_rewards.ipynb`: Exploring mixed intrinsic rewards;
    - `5 custom_intrinsic_reward.ipynb`: Custom intrinsic rewards.
2. `experiments`: Code for debugging and experiments.
    - `rllte/xplore/reward`: Implementations of the intrinsic rewards;
    - `scripts`: Training scripts for RQ 1-7;
    - `train_ppo.py`: Train the PPO agent with intrinsic rewards;
    - `train_ppo_lstm.py`: Train the PPO agent with intrinsic rewards and LSTM module;
    - `train_mixed.py`: Train the PPO agent with the mixed intrinsic rewards;
    - `mixed.py`: Create the mixed intrinsic rewards;
    - `train_cleanrl_sh.py`: Train the CleanRL's PPO agent with RLeXplore, for the *Montezuma's Revenge* game; 
    - `sac_cleanrl.py`: Train the SAC agent with intrinsic rewards, for the *Ant-Umaze* environment.