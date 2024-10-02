from utils import parse_args, make_env, select_intrinsic_reward
from rllte.agent import PPO, TwoHeadPPO

if __name__ == "__main__":
    # env setup
    args = parse_args()
    eval_args = parse_args()
    eval_args.n_envs = 1

    env, env_name = make_env(args, args.device)
    #eval_env, _ = make_env(eval_args, args.device)
    
    # select intrinsic reward
    intrinsic_reward = select_intrinsic_reward(
        args=args,
        env=env,
        device=args.device,
    )
    
    exp_name = f"{args.rq}_{'twoHeadPPO' if args.two_head else 'PPO'}_{env_name}_{args.intrinsic_reward}_obsRMS:{args.obs_rms}_rewNorm:{args.rwd_norm_type}_updateProp:{args.update_proportion}_rff:{args.int_gamma is not None}_weightInit:{args.weight_init}_s{args.seed}"

    # create agent
    if args.two_head:
        ppo_args = dict(
            env=env, 
            #eval_env=eval_env,
            seed=args.seed,
            device=args.device,
            tag=exp_name,
            hidden_dim=args.hidden_dim,
            feature_dim=args.feature_dim,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eps=args.eps,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            discount=args.discount,
            int_discount=args.discount,
            init_fn=args.init_fn,
            pretraining=False,
            encoder_model="espeholt",
        )
    else:
        ppo_args = dict(
            env=env, 
            #eval_env=eval_env,
            seed=args.seed,
            device=args.device,
            tag=exp_name,
            hidden_dim=args.hidden_dim,
            feature_dim=args.feature_dim,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eps=args.eps,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            discount=args.discount,
            init_fn=args.init_fn,
            pretraining=args.pretraining,
            gae_lambda=args.gae_lambda,
            encoder_model="espeholt",
        )
    
    if args.two_head:
        agent_class = TwoHeadPPO
    else:
        agent_class = PPO
        
    agent = agent_class(**ppo_args)
        
    # create intrinsic reward
    if intrinsic_reward is not None:
        agent.set(reward=intrinsic_reward)
        
    print("==== AGENT ====")
    print(agent.encoder)
    print(agent.policy)

    # start training
    agent.train(
        num_train_steps=args.num_train_steps,
        anneal_lr=args.anneal_lr,
        eval_interval=500,
        save_interval=999_999_999,
        th_compile=False
    )