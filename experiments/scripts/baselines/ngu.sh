for seed in 1 2 3 4 5;
do
	python train_ppo.py --pretraining \
	    --intrinsic_reward=ngu \
	    --env_id="SuperMarioBros-1-1-v3" \
	    --seed=${seed} \
	    --rq=baselines --device=cuda:1;
done
