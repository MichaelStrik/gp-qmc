from matplotlib import pyplot as plt
plt.style.use('seaborn-v0_8-paper')
import numpy as np
import pickle

from experiments import read_experiment_data


def plot_experiment(experiment):
    # read out setup
    Ndesign_list = np.array(experiment['Ndesign_list'])
    dim_list = experiment['dim_list']

    error_data = experiment['error_data']
    conditioning_data = experiment['conditioning_data']

    weights_decay = experiment.get('weights_decay', 2.5)
    type_design_points = experiment['type_design_points']


    fig, (ax1, ax2) = plt.subplots(1,2)

    # errors
    for i in range(error_data.shape[0]):
        ax1.loglog(Ndesign_list, error_data[i,:], 'o-', label=f's={dim_list[i]}')
    # add N^-1 as reference
    ax1.loglog(Ndesign_list, 1/Ndesign_list, linestyle='--', color='k', label=r'$N^{-1}$')

    # condition numbers
    if conditioning_data is not None:
        for i in range(error_data.shape[0]):
            ax2.loglog(Ndesign_list, conditioning_data[i,:], 'o-', label=f's={dim_list[i]}')

    # labels and co.
    ax1.set(xlabel=f'N ({type_design_points} design points)', ylabel='error (shift-average L2)')
    ax2.set(xlabel=f'N ({type_design_points} design points)', ylabel='condition number (average)')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    ax1.grid(which="minor", color="0.9")
    ax2.grid(which="minor", color="0.9")
    fig.suptitle(rf'GP/Kernel interpolant; $\gamma_j = 1/j^{{{weights_decay}}}$. Kernel: {experiment['kernel_class'].__name__}')
    plt.show()


if __name__ ==  '__main__':
    # define behavior as a command-line tool
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("plot_fun", help="Experiment function to be run.")
    parser.add_argument("exp_obj", help="Path to pickle file containing experiment data.")
    args = parser.parse_args()

    plot_function = locals().get(args.plot_fun)
    with open(args.exp_obj, 'rb') as file:
        experiment_object = pickle.load(file)
    
    plot_function(experiment_object)
