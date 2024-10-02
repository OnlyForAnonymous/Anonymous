from rllte.xplore.reward import *
import gymnasium as new_gym
import numpy as np

new_obs_space = new_gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
new_act_space = new_gym.spaces.Discrete(18)

def get_irs(args):
    if args.exp_name == 'rnd':
        irs = RND(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=512,
        update_proportion=args.update_proportion,
        encoder_model = "mnih",
        weight_init = "orthogonal")
    if args.exp_name == 're3':
        irs = RE3(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        latent_dim=256,
        encoder_model = "mnih",
        weight_init = "orthogonal"
    )
    if args.exp_name == 'ngu':
        irs = NGU(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=512,
        update_proportion=args.update_proportion,
        encoder_model = "mnih",
        weight_init = "orthogonal")   
    
    if args.exp_name == 'ride':
        irs = RIDE(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=512,
        update_proportion=args.update_proportion,
        encoder_model = "mnih",
        weight_init = "orthogonal") 
    
    if args.exp_name == 'icm':
        irs = ICM(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=512,
        update_proportion=args.update_proportion,
        encoder_model = "mnih",
        weight_init = "orthogonal") 
    
    if args.exp_name == 'dis':
        irs = Disagreement(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=512,
        update_proportion=args.update_proportion,
        encoder_model = "mnih",
        weight_init = "orthogonal") 
    
    if args.exp_name == 'e3b':
        irs = E3B(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=512,
        update_proportion=args.update_proportion,
        encoder_model = "mnih",
        weight_init = "orthogonal") 
    
    if args.exp_name == 'pc':
        irs = PseudoCounts(
        observation_space=new_obs_space,
        action_space=new_act_space,
        n_envs=args.num_envs,
        device=args.device,
        rwd_norm_type='rms',
        obs_rms=True,
        gamma=args.int_gamma,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        latent_dim=512,
        update_proportion=args.update_proportion,
        encoder_model = "mnih",
        weight_init = "orthogonal") 
    
    return irs