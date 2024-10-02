for candidate in 0.5 0.1 0.01;
    for algo in icm disagreement rnd ngu pseudocounts ride re3 e3b;
    do
        for seed in 1 2 3 4 5 6 7 8 9 10;
        do
            python train_ppo.py --pretraining \
                --weight_init=orthogonal \
                --obs_rms \
                --update_proportion=${candidate} \
                --rwd_norm_type=rms \
                --intrinsic_reward=${algo} \
                --env_id="SuperMarioBros-1-1-v3" \
                --rq=q3 \
                --seed=${seed}
        done
    done
done