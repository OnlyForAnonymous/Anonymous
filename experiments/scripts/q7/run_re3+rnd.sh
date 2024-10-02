for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=re3+rnd \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/re3+rnd_output_${seed}.log 2>&1 &
done

wait