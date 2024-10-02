for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=e3b+icm \
        --device cuda:0 \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/e3b+icm_output_${seed}.log 2>&1 &
done

for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=e3b+ride \
        --device cuda:1 \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/e3b+ride_output_${seed}.log 2>&1 &
done

for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=e3b+rnd \
        --device cuda:2 \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/e3b+rnd_output_${seed}.log 2>&1 &
done

wait