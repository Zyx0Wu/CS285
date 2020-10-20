import os
import sys
import time
import numpy as np
from collections import OrderedDict

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure import utils
from cs285.infrastructure.dqn_utils import get_env_kwargs, get_wrapper_by_name

save_dir = Path(__file__).absolute().parents[2] / 'figs'

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


def run_training_loop(self, n_iter, collect_policy, eval_policy,
                      initial_expertdata=None, relabel_with_expert=False,
                      start_relabel_with_expert=1, expert_policy=None):
    """
    :param n_iter:  number of (dagger) iterations
    :param collect_policy:
    :param eval_policy:
    :param initial_expertdata:
    :param relabel_with_expert:  whether to perform dagger
    :param start_relabel_with_expert: iteration at which to start relabel with expert
    :param expert_policy:
    """

    # init vars at beginning of training
    self.total_envsteps = 0
    self.start_time = time.time()

    print_period = 1000 if isinstance(self.agent, DQNAgent) else 1

    train_returns = {"Train_AverageReturn": [],
                     "Train_BestReturn":    []}
    for itr in range(n_iter):
        if itr % print_period == 0:
            print("\n\n********** Iteration %i ************" % itr)

        # decide if videos should be rendered/logged at this iteration
        if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
            self.logvideo = True
        else:
            self.logvideo = False

        # decide if metrics should be logged
        if self.params['scalar_log_freq'] == -1:
            self.logmetrics = False
        elif itr % self.params['scalar_log_freq'] == 0:
            self.logmetrics = True
        else:
            self.logmetrics = False

        # collect trajectories, to be used for training
        if isinstance(self.agent, DQNAgent):
            # only perform an env step and add to replay buffer for DQN
            self.agent.step_env()
            envsteps_this_batch = 1
            train_video_paths = None
            paths = None
        else:
            use_batchsize = self.params['batch_size']
            if itr == 0:
                use_batchsize = self.params['batch_size_initial']
            paths, envsteps_this_batch, train_video_paths = (
                self.collect_training_trajectories(
                    itr, initial_expertdata, collect_policy, use_batchsize)
            )

        self.total_envsteps += envsteps_this_batch

        # relabel the collected obs with actions from a provided expert policy
        if relabel_with_expert and itr >= start_relabel_with_expert:
            paths = self.do_relabel_with_expert(expert_policy, paths)

        # add collected data to replay buffer
        self.agent.add_to_replay_buffer(paths)

        # train agent (using sampled data from replay buffer)
        if itr % print_period == 0:
            print("\nTraining agent...")
        all_logs = self.train_agent()

        # log/save
        if self.logvideo or self.logmetrics:
            # perform logging
            print('\nBeginning logging procedure...')
            if isinstance(self.agent, DQNAgent):
                average, best = self.perform_dqn_logging(all_logs)
                train_returns["Train_AverageReturn"].append(average)
                train_returns["Train_BestReturn"].append(best)
            else:
                self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

            if self.params['save_params']:
                self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    return train_returns

def perform_dqn_logging(self, all_logs):
    last_log = all_logs[-1]

    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
    if len(episode_rewards) > 0:
        self.mean_episode_reward = np.mean(episode_rewards[-100:])
    if len(episode_rewards) > 100:
        self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    logs = OrderedDict()

    logs["Train_EnvstepsSoFar"] = self.agent.t
    print("Timestep %d" % (self.agent.t,))
    logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
    print("mean reward (100 episodes) %f" % self.mean_episode_reward)
    logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
    print("best mean reward %f" % self.best_mean_episode_reward)

    if self.start_time is not None:
        time_since_start = (time.time() - self.start_time)
        print("running time %f" % time_since_start)
        logs["TimeSinceStart"] = time_since_start

    logs.update(last_log)

    sys.stdout.flush()

    for key, value in logs.items():
        print('{} : {}'.format(key, value))
        self.logger.log_scalar(value, key, self.agent.t)
    print('Done logging...\n\n')

    self.logger.flush()

    return logs["Train_AverageReturn"], logs["Train_BestReturn"]

RL_Trainer.run_training_loop = run_training_loop
RL_Trainer.perform_dqn_logging = perform_dqn_logging


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
        return self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )

def main():

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
    return trainer.run_training_loop()


if __name__ == "__main__":
    train_returns = main()

    _, ax = plt.subplots()
    for k, v in train_returns.items():
        ax.plot(np.arange(len(v)), v, label=k)
    ax.set_title("MsPacman-v0")
    ax.set_xlabel("Iteration")
    ax.set_ylabel('Reward')
    ax.legend()

    plt.savefig(save_dir / 'q1.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()

