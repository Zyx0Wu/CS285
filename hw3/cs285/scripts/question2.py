import os
import time

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs

from pathlib import Path
save_dir = Path(__file__).absolute().parents[2] / 'figs'

from cs285.scripts.read_results import get_section_results
def read_result(logdir, y_tag='Eval_AverageReturn', x_tag='Train_EnvstepsSoFar'):
    import glob

    eventfile = glob.glob(logdir)[0]
    X, Y = get_section_results(eventfile, y_tag=y_tag, x_tag=x_tag)

    return X, Y


class Q_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_kwargs(params['env_name'])

        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = DQNAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )

def main(seed=0, double_q=True):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='MsPacman-v0',
        choices=('PongNoFrameskip-v4', 'LunarLander-v3', 'MsPacman-v0')
    )

    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--video_log_freq', type=int, default=-1)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    args.seed = seed
    args.double_q = double_q
    args.exp_name = 'q2_' + ('double' if args.double_q else '') + 'dqn_' + str(args.seed)
    params = vars(args)
    params['video_log_freq'] = -1 # This param is not used for DQN
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = 'hw3_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Q_Trainer(params)
    trainer.run_training_loop()

    return args.exp_name, logdir


if __name__ == "__main__":
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    seeds = [1, 2, 3]
    double_q = [False, True]

    _, ax = plt.subplots()
    for d in double_q:
        logs = {}
        for s in seeds:
            exp_name, logdir = main(s,d)
            logs[exp_name] = logdir + '/events*'

        xs, ys = [], []
        for k, v in logs.items():
            x, y = read_result(v, y_tag='Train_AverageReturn')
            if len(x) > len(y):
                y = [float("nan")] + y
            xs.append(x), ys.append(y)
        xs, ys = np.array(xs), np.array(ys)

        x = xs.mean(axis=0)
        y_mean, y_std = ys.mean(axis=0), ys.std(axis=0)
        ax.plot(x, y_mean, label='q2_'+('double' if d else 'single'))
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_title("LunarLander-v3")
    ax.set_xlabel("Iteration")
    ax.set_ylabel('Reward')
    ax.legend()

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)

    plt.savefig(save_dir / 'q2.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()

