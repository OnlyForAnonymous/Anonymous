for seed in 1 2 3 4 5;
do
    python train_ppo.py --num_train_steps=25_000_000  --intrinsic_reward=icm --env_id=procgen_AllMazeHard --beta=0.1 --two_head  --rwd_norm_type=none --obs_rms --update_proportion=1.0 --seed=${seed};
done

for seed in 1 2 3 4 5;
do
    python train_ppo.py --num_train_steps=25_000_000  --intrinsic_reward=icm --env_id=procgen_AllMazeHard --beta=0.1  --rwd_norm_type=none --obs_rms --update_proportion=1.0 --seed=${seed};
done