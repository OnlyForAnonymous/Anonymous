for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=icm+ride \
        --device=cuda:0 \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/icm+ride_output_${seed}.log 2>&1 &
done

for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=icm+rnd \
        --device=cuda:1 \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/icm+rnd_output_${seed}.log 2>&1 &
done

for seed in 1 2 3 4 5;
do
	python train_mixed.py --pretraining \
		--intrinsic_reward=rnd+ride \
        --device=cuda:2 \
		--env_id=SuperMarioBrosRandomStages-v3 \
		--seed=${seed} > logs/rnd+ride_output_${seed}.log 2>&1 &
done

wait