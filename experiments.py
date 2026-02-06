import pickle

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-v0_8-paper')

from kernels import UnanchoredSobolevKernel, AnchoredSobolevKernel
from interpolant import KernelInterpolant
from testfunction import TestFunction


def interpolate(test_function, kernel, Ndesign, num_design_shifts, type_design_points='lattice'):
    dim = kernel.d
    test_function_norm = test_function.kernel_norm()

    # parameters for error approximation
    k = 9
    num_error_shifts = 10

    interpolants = []
    errors = np.zeros(num_design_shifts)
    for shift in range(num_design_shifts):
        # construct qmc points
        interpolant = KernelInterpolant(kernel, dim)
        interpolant.generate_design_points(Ndesign, method=type_design_points)

        # evaluate test function at design points
        xdesign = interpolant.xdesign
        interpolant.ydesign = test_function(xdesign)

        # build interpolant and approximate error
        interpolant.build_interpolant()
        errors[shift] = interpolant.error_L2squared(test_function, Nsamples=2**k, qmc_shifts=num_error_shifts)/test_function_norm
        interpolants.append(interpolant)

    return interpolants, errors


def run_experiment( kernel_class, 
                    dim_list, 
                    Ndesign_list, 
                    num_test_functions, 
                    num_anchors, 
                    num_design_shifts, 
                    type_design_points, 
                    weights_decay=2.5,
                    **kwargs
                    ):
    errors = np.zeros((len(dim_list), len(Ndesign_list)))
    condition_numbers = np.zeros((len(dim_list), len(Ndesign_list)))

    for idx_dim in range(len(dim_list)):
        dim = dim_list[idx_dim]
        gamma = [1/j**weights_decay for j in range(1,dim+1)]
        kernel = kernel_class(dim, lengthscales=gamma)
        test_function_list = []
        for i in range(num_test_functions):
            test_function_list.append(TestFunction(kernel, dim, num_anchors))

        # average over different test functions/problems
        for idx_Ndesign in range(len(Ndesign_list)):
            Ndesign = Ndesign_list[idx_Ndesign]
            error_shift_avg     = []
            condition_shift_avg = [] # for consistency, we compute an average. 
                                     # but actually we don't expect the condition number to change with shifts
            # for every test problem, compute an averaged error over different shifts
            for test_function in test_function_list:
                interpolants_design_shifts, errors_design_shifts = interpolate(test_function, kernel, Ndesign, num_design_shifts, type_design_points)
                error_shift_avg.append(np.sqrt(np.mean(errors_design_shifts)))
                condition_shift_avg.append(np.mean([interpolants_design_shifts[j].condition_number for j in range(num_design_shifts)]))
            error_testfun_avg       = np.mean(error_shift_avg)
            condition_testfun_avg   = np.mean(condition_shift_avg)
            errors[idx_dim, idx_Ndesign]            = error_testfun_avg
            condition_numbers[idx_dim, idx_Ndesign] = condition_testfun_avg

    return errors, condition_numbers


def write_experiment_data(experiment):
    # if os.path.isfile(filename):
    #     raise ValueError("File already exists")
    #     return
    filename  = experiment['name']

    with open(filename, "wb") as file:
        pickle.dump(experiment, file)


def read_experiment_data(filename):
    with open(filename, "rb") as file:
        saved_experiment = pickle.load(file)
    return saved_experiment


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



model_experiment = {
    'kernel_class':  UnanchoredSobolevKernel,
    'weights_decay':  2.5,
    'dim_list':  [25],
    'Ndesign_list':  [16, 32, 64, 128, 256, 512],
    'num_test_functions':  4,
    'num_anchors':  10,
    'num_design_shifts':  1,
    'type_design_points':  'lattice',
    'name':  'fast_experiment',
    'error_data':  None,
    'conditioning_data':  None,
}


anchored_experiment = {
    'name': 'anchored_experiment',
    'dim_list': [4, 25, 50],
    'num_test_functions': 4,
    'num_anchors': 10,
    'Ndesign_list': [16, 32, 64, 128, 256, 512],
    'type_design_points': 'lattice',
    'num_design_shifts': 5,
    'kernel_class': AnchoredSobolevKernel,
    'weights_decay':  2.5,
    'error_data': None,
    'conditioning_data': None,
    }


first_experiment = {
    'kernel_class': UnanchoredSobolevKernel,
    'dim_list': [2, 4, 8, 25],
    'Ndesign_list': [16, 32, 64, 128, 256, 512],
    'num_test_functions': 4,
    'num_anchors': 10,
    'num_design_shifts': 5,
    'type_design_points': 'lattice',
    'name': 'first_experiment',
    'error_data': None,
    'conditioning_data': None,
    }
    

experiment_d100 = {
    'kernel_class': UnanchoredSobolevKernel,
    'dim_list': [2, 4, 8, 25, 50, 100],
    'Ndesign_list': [16, 32, 64, 128, 256, 512],
    'num_test_functions': 4,
    'num_anchors': 10,
    'num_design_shifts': 5,
    'type_design_points': 'lattice',
    'name': 'experiment_d100',
    'error_data': None,
    'conditioning_data': None,
    }


experiment_conditioning = {
    'name': 'experiment_conditioning',

    'kernel_class': UnanchoredSobolevKernel,
    'weights_decay':  2.5,
    'type_design_points': 'lattice',
    'Ndesign_list': [16, 32, 64, 128, 256, 512],
    'num_design_shifts': 1,
    
    'dim_list': [4, 25, 100],
    'num_test_functions': 1,
    'num_anchors': 10,

    'error_data': None,
    'conditioning_data': None,
    }


experiment_mc = {
    'name': 'experiment_mc',
    
    'kernel_class': UnanchoredSobolevKernel,
    'weights_decay':  2.5,
    'type_design_points': 'mc',
    'Ndesign_list': [16, 32, 64, 128, 256, 512],
    'num_design_shifts': 1,
    
    'dim_list': [4, 25, 100],
    'num_test_functions': 1,
    'num_anchors': 10,
    
    'error_data': None,
    'conditioning_data': None,
    }


if __name__ ==  '__main__':
    # experiment = experiment_d100
    experiment = anchored_experiment
    # experiment = experiment_conditioning

    mode = 'load'

    if mode=='run':
        error_table, condition_table = run_experiment(**experiment)
        experiment['error_data'] = error_table
        experiment['conditioning_data'] = condition_table
        write_experiment_data(experiment)
    elif mode=='load':
        experiment = read_experiment_data(experiment['name'])

    plot_experiment(experiment)