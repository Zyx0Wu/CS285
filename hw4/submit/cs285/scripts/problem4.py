import os
import time

from pathlib import Path
save_dir = Path(__file__).absolute().parents[2] / 'figs'

from cs285.scripts.read_results import get_section_results
def read_result(logdir, y_tag='Eval_AverageReturn', x_tag='Train_EnvstepsSoFar'):
    import glob

    eventfile = glob.glob(logdir)[0]
    X, Y = get_section_results(eventfile, y_tag=y_tag, x_tag=x_tag)

    return X, Y


if __name__ == "__main__":
    import re
    import glob
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    logdirs = glob.glob('data/hw4_q4_reacher_horizon*')

    logs = {}
    for d in logdirs:
        exp_name = re.findall("horizon[0-9]*_reacher", d)[0]
        logs[exp_name] = d + '/events*'

    _, ax = plt.subplots()
    for k, v in logs.items():
        x, y = read_result(v, y_tag='Eval_AverageReturn')
        if len(x) > len(y):
            y = [float("nan")] + y
        ax.plot(x, y, label=k)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_title("Reacher-v0")
    ax.set_xlabel("Envsteps")
    ax.set_ylabel('Reward')
    ax.legend()

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)

    plt.savefig(save_dir / 'q4_horizon.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()

