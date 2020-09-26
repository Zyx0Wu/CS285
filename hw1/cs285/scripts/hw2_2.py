from collections import OrderedDict
import numpy as np
import os
import time

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.bc_agent import BCAgent
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs285.infrastructure import utils

save_dir = Path(__file__).absolute().parents[2] / 'figs'

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


def run_training_loop(self, n_iter, collect_policy, eval_policy,
                      initial_expertdata=None, relabel_with_expert=False,
                      start_relabel_with_expert=1, expert_policy=None):

    # init vars at beginning of training
    self.total_envsteps = 0
    self.start_time = time.time()

    eval_means, eval_stds = [], []
    for itr in range(n_iter):
        print("\n\n********** Iteration %i ************" % itr)

        # decide if videos should be rendered/logged at this iteration
        if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
            self.log_video = True
        else:
            self.log_video = False

        # decide if metrics should be logged
        if itr % self.params['scalar_log_freq'] == 0:
            self.log_metrics = True
        else:
            self.log_metrics = False

        # collect trajectories, to be used for training
        training_returns = self.collect_training_trajectories(
            itr,
            initial_expertdata,
            collect_policy,
            self.params['batch_size']
        )  # HW1: implement this function below
        paths, envsteps_this_batch, train_video_paths = training_returns
        self.total_envsteps += envsteps_this_batch

        # relabel the collected obs with actions from a provided expert policy
        if relabel_with_expert and itr >= start_relabel_with_expert:
            paths = self.do_relabel_with_expert(expert_policy, paths)  # HW1: implement this function below

        # add collected data to replay buffer
        self.agent.add_to_replay_buffer(paths)

        # train agent (using sampled data from replay buffer)
        training_logs = self.train_agent()  # HW1: implement this function below

        # log/save
        if self.log_video or self.log_metrics:

            # perform logging
            print('\nBeginning logging procedure...')
            eval_mean, eval_std = self.perform_logging(
                itr, paths, eval_policy, train_video_paths, training_logs)
            eval_means.append(eval_mean)
            eval_stds.append(eval_std)

            if self.params['save_params']:
                print('\nSaving agent params')
                self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))

    return (eval_means, eval_stds)

def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):
    # collect eval trajectories, for logging
    print("\nCollecting data for eval...")
    eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy,
                                                                     self.params['eval_batch_size'],
                                                                     self.params['ep_len'])

    # save eval rollouts as videos in tensorboard event file
    if self.log_video and train_video_paths != None:
        print('\nCollecting video rollouts eval')
        eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        # save train/eval videos
        print('\nSaving train rollouts as videos...')
        self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                        video_title='train_rollouts')
        self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                        video_title='eval_rollouts')

    # save eval metrics
    if self.log_metrics:
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
        last_log = training_logs[-1]  # Only use the last log for now
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

        return (logs["Eval_AverageReturn"], logs["Eval_StdReturn"])

RL_Trainer.run_training_loop = run_training_loop
RL_Trainer.perform_logging = perform_logging


class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        return self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )


def main(dagger=True, num_iters=1):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    args.do_dagger = dagger
    args.n_iter = num_iters
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################

    trainer = BC_Trainer(params)
    return trainer.run_training_loop()

if __name__ == "__main__":
    iters = 10
    da_eval_means, da_eval_stds = main(True, iters)

    bc_eval_mean, bc_eval_std = main(False)
    bc_eval_means = bc_eval_mean * iters
    bc_eval_stds = bc_eval_std * iters

    expert_means = [3772.67] * iters
    expert_stds = [0.00] * iters

    plt.figure()
    plt.errorbar(range(iters), bc_eval_means, bc_eval_stds, marker='o', capsize=8, linestyle='--', label='Behavior Cloning')
    plt.errorbar(range(iters), da_eval_means, da_eval_stds, marker='o', capsize=8, linestyle='--', label='DAgger')
    plt.errorbar(range(iters), expert_means, expert_stds, marker='o', capsize=8, linestyle='--', label='Expert')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Reward')
    plt.legend()

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    plt.savefig(save_dir / 'bc2_2_2.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()

