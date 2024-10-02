for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=icm+ride \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/icm+ride_output_${seed}.log 2>&1 &
done

wait