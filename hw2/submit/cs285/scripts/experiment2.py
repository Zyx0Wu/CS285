import os
import time
import itertools
import numpy as np
from collections import OrderedDict

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.pg_agent import PGAgent
from cs285.infrastructure import utils

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

    eval_means = []
    for itr in range(n_iter):
        print("\n\n********** Iteration %i ************" % itr)

        # decide if videos should be rendered/logged at this iteration
        if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
            self.logvideo = True
        else:
            self.logvideo = False
        self.log_video = self.logvideo

        # decide if metrics should be logged
        if self.params['scalar_log_freq'] == -1:
            self.logmetrics = False
        elif itr % self.params['scalar_log_freq'] == 0:
            self.logmetrics = True
        else:
            self.logmetrics = False

        # collect trajectories, to be used for training
        training_returns = self.collect_training_trajectories(itr,
                                                              initial_expertdata, collect_policy,
                                                              self.params['batch_size'])
        paths, envsteps_this_batch, train_video_paths = training_returns
        self.total_envsteps += envsteps_this_batch

        # add collected data to replay buffer
        self.agent.add_to_replay_buffer(paths)

        # train agent (using sampled data from replay buffer)
        train_logs = self.train_agent()

        # log/save
        if self.logvideo or self.logmetrics:
            # perform logging
            print('\nBeginning logging procedure...')
            eval_means.append(self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs))

            if self.params['save_params']:
                self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    return eval_means

def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):
    last_log = all_logs[-1]

    #######################

    # collect eval trajectories, for logging
    print("\nCollecting data for eval...")
    eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy,
                                                                     self.params['eval_batch_size'],
                                                                     self.params['ep_len'])

    # save eval rollouts as videos in tensorboard event file
    if self.logvideo and train_video_paths != None:
        print('\nCollecting video rollouts eval')
        eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        # save train/eval videos
        print('\nSaving train rollouts as videos...')
        self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                        video_title='train_rollouts')
        self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                        video_title='eval_rollouts')

    #######################

    # save eval metrics
    if self.logmetrics:
        # returns, for logging
        train_returns = [path["reward"].sum() for path in paths]
        eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

        # episode lengths, for logging
        train_ep_lens = [len(path["reward"]) for path in paths]
        eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        # decide what to log
        logs = OrderedDict()
        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

        logs["Train_AverageReturn"] = np.mean(train_returns)
        logs["Train_StdReturn"] = np.std(train_returns)
        logs["Train_MaxReturn"] = np.max(train_returns)
        logs["Train_MinReturn"] = np.min(train_returns)
        logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

        logs["Train_EnvstepsSoFar"] = self.total_envsteps
        logs["TimeSinceStart"] = time.time() - self.start_time
        logs.update(last_log)

        if itr == 0:
            self.initial_return = np.mean(train_returns)
        logs["Initial_DataCollection_AverageReturn"] = self.initial_return

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        print('Done logging...\n\n')

        self.logger.flush()

        return logs["Eval_AverageReturn"]

RL_Trainer.run_training_loop = run_training_loop
RL_Trainer.perform_logging = perform_logging


class PG_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        return self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            )


def main(batch_size=1000, learning_rate=5e-3):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    args.batch_size = batch_size
    args.learning_rate = learning_rate
    args.exp_name = "q2" + "_b"+str(args.batch_size) + "_r"+str(args.learning_rate)
    params = vars(args)

    ## ensure compatibility with hw1 code
    params['train_batch_size'] = params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    trainer = PG_Trainer(params)
    return trainer.run_training_loop()


if __name__ == "__main__":

    # search
    eval_dict = {}
    batch_size = [40, 60, 80]
    learning_rate = [2e-2, 4e-2, 6e-2]
    for b,r in itertools.product(batch_size, learning_rate):
        eval_means = main(b, r)
        eval_dict["q2" + "_b"+str(b) + "_r"+str(r)] = [b, r, max(eval_means)]

    fig, ax = plt.subplots()
    for k,v in eval_dict.items():
        ax.scatter(v[0], v[1], s=v[2])
        ax.annotate(v[2], (v[0], v[1]))
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: "{:.1e}".format(x)))
    ax.set_title("InvertedPendulum-v2")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel('Learning Rate')

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    fig.savefig(save_dir / 'q2_search.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close(fig)

    # optimal
    eval_dict = {}
    b, r = (60, 4e-2)
    eval_means = main(b, r)
    eval_dict["q2" + "_b"+str(b) + "_r"+str(r)] = eval_means

    plt.figure()
    for k, v in eval_dict.items():
        plt.plot(np.arange(len(v)) + 1, v, label=k)
    plt.title("InvertedPendulum-v2")
    plt.xlabel("Iteration")
    plt.ylabel('Reward')
    plt.legend()

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    plt.savefig(save_dir / 'q2_opt.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()

