import pickle
import numpy as np

from kernels import UnanchoredSobolevKernel, AnchoredSobolevKernel
from interpolant import KernelInterpolant
from testproblems import TestFunction, EllipticProblem

from numbers import Number


def interpolate(test_function, kernel, Ndesign, num_design_shifts, type_design_points='lattice'):
    dim = kernel.d
    if type(test_function) is TestFunction:
        # for the artificial problem, the kernel norm is cheap to compute
        test_function_norm = test_function.kernel_norm()
    else:
        # otherwise, we leave normalisation to the error computing function
        test_function_norm = None

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
        errors[shift] = interpolant.error_L2squared(test_function, Nsamples=2**k, qmc_shifts=num_error_shifts, normalisation=test_function_norm) # TODO Is normalisation by the squared norm more appropriate?
        interpolants.append(interpolant)

    return interpolants, errors


def run_bvp_experiment( kernel_class,
                    dim_list,
                    Ndesign_list,
                    num_design_shifts,
                    type_design_points,
                    weights_decay=2.5,
                    **kwargs
                    ):
    """
    weights_decay: Number or tuple of two numbers, in which case the first number specifies the decay of weights in the problem and the second the decay in the method.
    """
    errors = np.zeros((len(dim_list), len(Ndesign_list)))
    condition_numbers = np.zeros((len(dim_list), len(Ndesign_list)))

    if isinstance(weights_decay, Number):
        weights_decay = (weights_decay, weights_decay)
    elif type(weights_decay) not in [tuple, list, np.array]:
        TypeError('weights_decay has to be a number or a tuple/list/np.array of two numbers')

    for idx_dim, dim in enumerate(dim_list):
        print(f"Dimension {dim}")
        gamma = [1/j**weights_decay[1] for j in range(1,dim+1)]
        kernel = kernel_class(dim, lengthscales=gamma)

        elliptic_bvp = EllipticProblem(dim, q=weights_decay[0])
        for idx_Ndesign, Ndesign in enumerate(Ndesign_list):
            print(f"\tNdesign {Ndesign}")
            interpolants_design_shifts, errors_design_shifts = interpolate(elliptic_bvp, kernel, Ndesign, num_design_shifts, type_design_points)
            errors[idx_dim, idx_Ndesign]            = np.sqrt(np.mean(errors_design_shifts)) # shift-average error
            condition_numbers[idx_dim, idx_Ndesign] = np.mean([interpolants_design_shifts[j].condition_number for j in range(num_design_shifts)]) # shift-average condition number

    return errors, condition_numbers


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

    for idx_dim, dim in enumerate(dim_list):
        gamma = [1/j**weights_decay for j in range(1,dim+1)]
        kernel = kernel_class(dim, lengthscales=gamma)
        test_function_list = []
        for i in range(num_test_functions):
            test_function_list.append(TestFunction(kernel, dim, num_anchors))

        # average over different test functions/problems
        for idx_Ndesign, Ndesign in enumerate(Ndesign_list):    
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


def run_condition_experiment(kernel_class, dim_list, Ndesign_list, num_design_shifts, type_design_points, weights_decay=2.5, **kwargs):
    condition_numbers = np.zeros((len(dim_list), len(Ndesign_list)))

    for idx_dim, dim in enumerate(dim_list):
        print(f"Dimension {dim}")

        for idx_Ndesign, Ndesign in enumerate(Ndesign_list):
            print(f"\tNdesign {Ndesign}")
            gamma = [1/j**weights_decay for j in range(1,dim+1)]
            kernel = kernel_class(dim, lengthscales=gamma)

            interpolant = KernelInterpolant(kernel, dim)
            interpolant.generate_design_points(Ndesign, method=type_design_points)
            K = interpolant.kernel(interpolant.xdesign[:,np.newaxis,:], interpolant.xdesign[np.newaxis,:,:])
            condition_numbers[idx_dim, idx_Ndesign] = np.linalg.cond(K)
            # print(f"Condition number: {interpolant.condition_number}")
    
    errors = []

    return errors, condition_numbers


def write_experiment_data(experiment):
    # if os.path.isfile(filename):
    #     raise ValueError("File already exists")
    #     return
    filename  = experiment['name']

    with open('./exp_data/'+filename, "wb") as file:
        pickle.dump(experiment, file)


def read_experiment_data(filename):
    with open(filename, "rb") as file:
        saved_experiment = pickle.load(file)
    return saved_experiment


if __name__ ==  '__main__':
    # define behavior as a command-line tool
    import argparse
    import json
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_fun", help="Experiment function to be run.")
    parser.add_argument("exp_obj", help="Path to json file containing an experiment dictionary to be passed to exp_fun.")
    args = parser.parse_args()
    experiment_function = locals().get(args.exp_fun)
    with open(args.exp_obj, 'r') as file:
        experiment_object = json.load(file)
    # replace kernel name by corresponding python class
    if type(experiment_object['kernel_class']) is str:
        kernel_dict = {'UnanchoredSobolevKernel':UnanchoredSobolevKernel, 'AnchoredSobolevKernel':AnchoredSobolevKernel}
        kernel_name = experiment_object['kernel_class']
        experiment_object['kernel_class'] = kernel_dict[kernel_name]

    t1 = time.time()
    error_table, condition_table = experiment_function(**experiment_object)
    duration = time.time() - t1
    print(f"Total runtime: {duration//(60*60*24):.0f}d {duration//(60*60)%24:2.0f}h {duration//(60)%60:2.0f}m {duration%60:2.0f}s")

    experiment_object['error_data'] = error_table
    experiment_object['conditioning_data'] = condition_table
    write_experiment_data(experiment_object)