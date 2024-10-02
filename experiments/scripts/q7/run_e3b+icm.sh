for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=e3b+icm \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/e3b+icm_output_${seed}.log 2>&1 &
done

wait