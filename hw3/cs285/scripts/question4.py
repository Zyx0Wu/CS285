import os
import time

from cs285.agents.ac_agent import ACAgent
from cs285.infrastructure.rl_trainer import RL_Trainer

from pathlib import Path
save_dir = Path(__file__).absolute().parents[2] / 'figs'

from cs285.scripts.read_results import get_section_results
def read_result(logdir, y_tag='Eval_AverageReturn', x_tag='Train_EnvstepsSoFar'):
    import glob

    eventfile = glob.glob(logdir)[0]
    X, Y = get_section_results(eventfile, y_tag=y_tag, x_tag=x_tag)

    return X, Y


class AC_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'num_target_updates': params['num_target_updates'],
            'num_grad_steps_per_target_update': params['num_grad_steps_per_target_update'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = ACAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            )


def main(num_target_updates=10, num_grad_steps_per_target_update=10):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=1)

    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=1000) ##steps used per gradient step

    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=10)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    args.num_target_updates = num_target_updates
    args.num_grad_steps_per_target_update = num_grad_steps_per_target_update
    args.exp_name = 'q4_' + str(num_target_updates) + '_' + str(num_grad_steps_per_target_update)
    params = vars(args)

    # for policy gradient, we made a design decision
    # to force batch_size = train_batch_size
    # note that, to avoid confusion, you don't even have a train_batch_size argument anymore (above)
    params['train_batch_size'] = params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = 'hw3_ ' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = AC_Trainer(params)
    trainer.run_training_loop()

    return args.exp_name, logdir


if __name__ == "__main__":
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    settings = [(1,1), (100,1), (1,100), (10,10)]

    logs = {}
    for t,g in settings:
        exp_name, logdir = main(t,g)
        logs[exp_name] = logdir + '/events*'

    _, ax = plt.subplots()
    for k, v in logs.items():
        x, y = read_result(v, y_tag='Eval_AverageReturn')
        if len(x) > len(y):
            y = [float("nan")] + y
        ax.plot(x, y, label=k)
    ax.set_title("CartPole-v0")
    ax.set_xlabel("Iteration")
    ax.set_ylabel('Reward')
    ax.legend()

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)

    plt.savefig(save_dir / 'q4.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()

