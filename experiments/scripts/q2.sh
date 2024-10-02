for candidate in minmax none;
    for algo in icm disagreement rnd ngu pseudocounts ride re3 e3b;
    do
        for seed in 1 2 3 4 5 6 7 8 9 10;
        do
            python train_ppo.py --pretraining \
                --weight_init=orthogonal \
                --obs_rms \
                --update_proportion=1.0 \
                --rwd_norm_type=${candidate} \
                --intrinsic_reward=${algo} \
                --env_id="SuperMarioBros-1-1-v3" \
                --rq=q2 \
                --seed=${seed}
        done
    done
done