for seed in 1 2 3 4 5;
do
    python train_ppo.py --num_train_steps=25_000_000  --intrinsic_reward=pseudocounts --env_id=procgen_1MazeHard --beta=0.1 --two_head  --rwd_norm_type=rms --obs_rms --update_proportion=0.5 --seed=${seed};
done

for seed in 1 2 3 4 5;
do
    python train_ppo.py --num_train_steps=25_000_000  --intrinsic_reward=pseudocounts --env_id=procgen_1MazeHard --beta=0.1  --rwd_norm_type=rms --obs_rms --update_proportion=0.5 --seed=${seed};
done