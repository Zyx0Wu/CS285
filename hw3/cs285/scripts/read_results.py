import glob
import tensorflow as tf

def get_section_results(file, y_tag='Eval_AverageReturn', x_tag='Train_EnvstepsSoFar'):
    X = []
    Y = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == x_tag:
                X.append(v.simple_value)
            elif v.tag == y_tag:
                Y.append(v.simple_value)
    return X, Y

if __name__ == '__main__':
    import glob

    logdir = 'data/q1_lb_rtg_na_CartPole-v0_13-09-2020_23-32-10/events*'
    eventfile = glob.glob(logdir)[0]

    X, Y = get_section_results(eventfile)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
