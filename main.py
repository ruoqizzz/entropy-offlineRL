import argparse
import gym
import numpy as np
import os
import torch
import json

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter

hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,    'max_q_backup': False,    'reward_tune': 'no',          'num_epochs': 2000, 'gn': 4.0, 'loss_type': 'MLL', 'action_clip': False},
    
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 2.0,    'max_q_backup': False,    'reward_tune': 'cql_antmaze', 'num_epochs': 1000, 'gn': 4.0, 'loss_type': 'NML', 'action_clip': True},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,    'max_q_backup': True,     'reward_tune': 'cql_antmaze', 'num_epochs': 1000, 'gn': 4.0, 'loss_type': 'NML', 'action_clip': True},
    'antmaze-medium-play-v0':        {'lr': 3e-4, 'eta': 2.0,    'max_q_backup': True,     'reward_tune': 'cql_antmaze', 'num_epochs': 1000, 'gn': 4.0, 'loss_type': 'NML', 'action_clip': True},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 2.0,    'max_q_backup': True,     'reward_tune': 'cql_antmaze', 'num_epochs': 1000, 'gn': 4.0, 'loss_type': 'NML', 'action_clip': True},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 2.0,    'max_q_backup': True,     'reward_tune': 'cql_antmaze', 'num_epochs': 1000, 'gn': 4.0, 'loss_type': 'NML', 'action_clip': True},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,    'max_q_backup': True,     'reward_tune': 'cql_antmaze', 'num_epochs': 1000, 'gn': 4.0, 'loss_type': 'NML', 'action_clip': True},
    
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.1,    'max_q_backup': False,    'reward_tune': 'normalize',   'num_epochs': 1000, 'gn': 8.0, 'loss_type': 'NML', 'action_clip': True},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,    'max_q_backup': False,    'reward_tune': 'normalize',   'num_epochs': 1000, 'gn': 8.0, 'loss_type': 'NML', 'action_clip': True},
    
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005,   'max_q_backup': False,    'reward_tune': 'no',         'num_epochs': 1000, 'gn': 10., 'loss_type': 'MLL', 'action_clip': False},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005,   'max_q_backup': False,    'reward_tune': 'no',         'num_epochs': 1000, 'gn': 10., 'loss_type': 'MLL', 'action_clip': False},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005,   'max_q_backup': False,    'reward_tune': 'no',         'num_epochs': 1000, 'gn': 10., 'loss_type': 'MLL', 'action_clip': False},
}

def train_agent(env, state_dim, action_dim, max_action, output_dir, args):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, args.device, args.reward_tune)
    utils.print_banner('Loaded buffer')

    if args.algo == 'eql':
        from agents.eql_diffusion import Diffusion_EQL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=args.device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      schedule=args.schedule,
                      n_timesteps=args.T,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn,
                      num_critics=args.num_critics,
                      pess_method=args.pess,
                      ent_coef=args.ent_coef,
                      lcb_coef=args.lcb_coef,
                      loss_type=args.loss_type,
                      action_clip=args.action_clip)
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=args.device,
                      discount=args.discount,
                      tau=args.tau,
                      schedule=args.schedule,
                      n_timesteps=args.T,
                      lr=args.lr,
                      loss_type=args.loss_type,
                      action_clip=args.action_clip)

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.record_tabular('Entropy', np.mean(loss_metric['entropy']))
        if 'ent_coef' in loss_metric.keys():
            logger.record_tabular('Entropy Coef', np.mean(loss_metric['ent_coef']))
        if 'ent_coef_loss' in loss_metric.keys():
            logger.record_tabular('Entropy Coef Loss', np.mean(loss_metric['ent_coef_loss']))
        logger.dump_tabular()


        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']),
                            np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']),
                            curr_epoch])
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.dump_tabular()

        bc_loss = np.mean(loss_metric['bc_loss'])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)

    best_id = np.argmax(scores[:, 2])
    best_res = {'model selection': args.algo, 'epoch': scores[best_id, -1],
                'best normalized score avg': scores[best_id, 2],
                'best normalized score std': scores[best_id, 3],
                'best raw score avg': scores[best_id, 0],
                'best raw score std': scores[best_id, 1]}

    with open(os.path.join(output_dir, f"best_score_{args.algo}.txt"), 'w') as f:
        f.write(json.dumps(best_res))

    # writer.close()


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="walker2d-medium-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--schedule", default='cosine', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="eql", type=str)  # ['bc', 'eql']

    parser.add_argument("--pess", default='lcb', type=str, help="['min', 'lcb']")
    parser.add_argument("--num_critics", default=64, type=int)
    parser.add_argument("--ent_coef", default=0.01, type=float)
    parser.add_argument("--lcb_coef", default=4, type=float)

    # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--eta", default=1.0, type=float)
    # parser.add_argument("--reward_tune", default='no', type=str)
    # parser.add_argument("--gn", default=-1.0, type=float)

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}/{args.algo}'

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.loss_type = hyperparameters[args.env_name]['loss_type']
    args.action_clip = hyperparameters[args.env_name]['action_clip']

    # Setup Logging
    file_name_prex = f"{args.env_name}|{args.exp}|"
    file_name = f"diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|{args.seed}'
    file_name += f'|ent_coef-{args.ent_coef}'

    if 'eql' in args.algo:
        file_name += f'|pess-{args.pess}-{args.lcb_coef}'
        file_name += f'|num_critics-{args.num_critics}'

    results_dir = os.path.join(args.output_dir, file_name_prex+file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"EnsembleQ-Diffusion-Policies-RL")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")


    train_agent(env, state_dim, action_dim, max_action, results_dir, args)
