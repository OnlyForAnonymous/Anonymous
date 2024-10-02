for seed in 1 2 3 4 5;
do
    python train_ppo.py --num_train_steps=25_000_000  --intrinsic_reward=extrinsic --env_id=procgen_AllMazeHard --seed=${seed};
done