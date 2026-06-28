import pickle
from numbers import Number
from collections.abc import Callable
from os import cpu_count

from multiprocess import Process, Queue
import numpy as np
import qmcpy

from kernels import UnanchoredSobolevKernel, AnchoredSobolevKernel
from interpolant import KernelInterpolant
from testproblems import TestFunction, EllipticProblemQOI, EllipticProblemFE
from utils import worker


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


def interpolate_multivar(test_function, output_dim, kernel, Ndesign, num_design_shifts, type_design_points='lattice'):
    input_dim = kernel.d
    interpolants = np.empty((num_design_shifts, output_dim), dtype=KernelInterpolant)
    for shift in range(num_design_shifts):
        # generate design points
        interpolant = KernelInterpolant(kernel, input_dim) # this interpolant is only needed to generate a common design
        interpolant.generate_design_points(Ndesign, method=type_design_points)
        xdesign = interpolant.xdesign
        ydesign = test_function(xdesign)

        for outdim in range(output_dim):
            interpolant = KernelInterpolant(kernel, input_dim)
            interpolant.xdesign = xdesign
            interpolant.ydesign = ydesign[:,outdim]
            interpolant.build_interpolant()
            interpolants[shift, outdim] = interpolant

    return interpolants


def L2errorFEM(interpolants: np.array, Nsamples, Nx, f:Callable, q, method='lattice', qmc_shifts=1, **kwargs):
    r"""
    Computes the error \int_U \int_D (u_h(x,y) - u_{h,n}(x,y))^2 dx dy over the physical domain D \subset \R and stocahstic domain U
    for FEM solutions u_h and kernel interpolants u_{h,n}.

    Args:
        interpolants:np.array
        Collection of interpolants that together represent a FEM interpolation. 
        A fixed mesh is assumed and there should be one interpolant for each degree of freedom.
        
        Nsamples : int
        Number of samples.

        Nx : int 
        Degree of freedoms of finite element method.

        f : Callable
        PDE right-hand-side

        q : float
        Decay rate of problem weights in parametric PDE.

        method : str
        Method. One of 'lattice', 'digital_net', 'MC' (Monte Carlo).

        qmc_shifts : int
        Number of quadrature shifts. Should not be used with method 'MC'.
    """
    
    ydim = interpolants[0].dim
    assert(len(interpolants) == Nx)

    # sample points
    method = method.lower()
    assert method in ['mc', 'lattice', 'digital_net']
    if method == 'mc':
        rng = np.random.default_rng()
        point_sets = rng.uniform(0, 1, size=(Nsamples, ydim))
        point_sets = point_sets[np.newaxis,:,:]
        qmc_shifts = 1
    elif method == 'lattice' or method == 'digital_net':
        qmcGens = {'lattice': qmcpy.Lattice, 'digital_net': qmcpy.DigitalNetB2}
        qmcGen = qmcGens[method](dimension=ydim, replications=qmc_shifts, **kwargs)
        point_sets = qmcGen(Nsamples, warn=False)
    # compute qmc_shifts many (shifted qmc) quadrature rules
    Qshifted = np.zeros(qmc_shifts)
    def worker_L2error_to_fem(input, output, ydim, Nx, f, q):
            from mpi4py import MPI
            from dolfinx import fem
            import ufl
            elliptic_problem = EllipticProblemFE(ydim, Nx, f, q)
            for pts, coefs in iter(input.get, 'STOP'):
                assert(pts.shape[0] == coefs.shape[0])
                Q_local = 0
                for y, coef_interp in zip(pts,coefs):
                    u_h = elliptic_problem.solve(y).u
                    u_hs    = fem.Function(elliptic_problem.V) # surrogate/interpolant
                    u_hs.x.array[:] = coef_interp
                    L2error = fem.form(ufl.inner(u_h-u_hs,u_h-u_hs)*ufl.dx)
                    error_local = fem.assemble_scalar(L2error)
                    L2error = np.sqrt(elliptic_problem.domain.comm.allreduce(error_local, op=MPI.SUM))
                    Q_local += L2error
                output.put(Q_local)
    N_PROCESSES = cpu_count()
    task_queue = Queue()
    done_queue = Queue()
    for shift in range(qmc_shifts):
        points = point_sets[shift,:,:]
        points_splits = np.array_split(points, N_PROCESSES)
        coefficients_splits = []
        # calculate interpolated fem coefficients at error sampling points
        for pts in points_splits:
            coefs = np.empty((pts.shape[0],Nx))
            for row, y in enumerate(pts):
                coefs[row,:] = np.concatenate([interpolant(y) for interpolant in interpolants])
            coefficients_splits.append(coefs)
        TASKS = [(pts,coefs) for (pts,coefs) in zip(points_splits,coefficients_splits)]
        for task in TASKS:
            task_queue.put(task)
        for n in range(N_PROCESSES):
            Process(target=worker_L2error_to_fem, args=(task_queue, done_queue, ydim, Nx, f, q)).start()
        for task in TASKS:
            Qshifted[shift] += done_queue.get()
        for n in range(N_PROCESSES):
            task_queue.put('STOP')
        Qshifted[shift] /= Nsamples
    
    return np.mean(Qshifted)


def run_bvp_experiment(exp_data):
    """
    weights_decay: Number or tuple of two numbers, in which case the first number specifies the decay of weights in the problem and the second the decay in the method.
    """
    # parameters
    kernel_class        = exp_data.get('kernel_class')
    dim_list            = exp_data.get('dim_list')
    Ndesign_list        = exp_data.get('Ndesign_list')
    num_design_shifts   = exp_data.get('num_design_shifts')
    type_design_points  = exp_data.get('type_design_points')
    weights_decay       = exp_data.get('weights_decay')

    # allocate
    errors = np.zeros((len(dim_list), len(Ndesign_list)))
    condition_numbers = np.zeros((len(dim_list), len(Ndesign_list)))

    if isinstance(weights_decay, Number):
        weights_decay = (weights_decay, weights_decay)
    elif type(weights_decay) not in [tuple, list, np.array]:
        TypeError('weights_decay has to be a number or a tuple/list/np.array of two numbers')

    # experiment
    for idx_dim, dim in enumerate(dim_list):
        print(f"Dimension {dim}")
        gamma = [1/j**weights_decay[1] for j in range(1,dim+1)]
        kernel = kernel_class(dim, lengthscales=gamma)

        elliptic_bvp = EllipticProblemQOI(dim, q=weights_decay[0])
        for idx_Ndesign, Ndesign in enumerate(Ndesign_list):
            print(f"\tNdesign {Ndesign}")
            interpolants_design_shifts, errors_design_shifts = interpolate(elliptic_bvp, kernel, Ndesign, num_design_shifts, type_design_points)
            errors[idx_dim, idx_Ndesign]            = np.sqrt(np.mean(errors_design_shifts)) # shift-average error
            condition_numbers[idx_dim, idx_Ndesign] = np.mean([interpolants_design_shifts[j].condition_number for j in range(num_design_shifts)]) # shift-average condition number

    exp_data['error_data'] = errors
    exp_data['conditioning_data'] = condition_numbers
    
    return exp_data


def run_experiment(exp_data):
    # parameters
    kernel_class        = exp_data.get('kernel_class')
    dim_list            = exp_data.get('dim_list')
    Ndesign_list        = exp_data.get('Ndesign_list')
    num_test_functions  = exp_data.get('num_test_functions')
    num_anchors         = exp_data.get('num_anchors')
    num_design_shifts   = exp_data.get('num_design_shifts')
    type_design_points  = exp_data.get('type_design_points')
    weights_decay       = exp_data.get('weights_decay')

    # allocate
    errors = np.zeros((len(dim_list), len(Ndesign_list)))
    condition_numbers = np.zeros((len(dim_list), len(Ndesign_list)))

    # experiment
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

    exp_data['error_data'] = errors
    exp_data['conditioning_data'] = condition_numbers
    
    return exp_data


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


def run_multiplication_experiment(exp_data):
    # parameters
    kernel_class = exp_data.get('kernel_class')
    dim_list = exp_data.get('dim_list')
    weights_decay = exp_data.get('weights_decay')
    num_test_functions = exp_data.get('num_test_functions')
    type_design_points = exp_data.get('type_design_points', 'lattice')
    Ndesign = exp_data.get('Ndesign')
    

    Nscaling_list = [10**N for N in range(5)]
    data = np.zeros((len(dim_list), num_test_functions))
    
    for idx_dim, dim in enumerate(dim_list):
        print(f"dim {dim}")
        # kernel
        gamma = [1/j**weights_decay for j in range(1,dim+1)]
        kernel = kernel_class(dim, lengthscales=gamma)
        
        # RKHS functions
        test_functions_list = []
        for i in range(num_test_functions):
            test_functions_list.append(TestFunction(kernel, dim, num_anchors=dim*10))

        for test_fct_idx, test_fct in enumerate(test_functions_list):
        # try a number of random functions
            # interpolate non-squared function
            interpolant = KernelInterpolant(kernel, dim)
            interpolant.generate_design_points(Ndesign, method=type_design_points)
            # sample and build
            xdesign = interpolant.xdesign
            interpolant.ydesign = test_fct(xdesign)
            interpolant.build_interpolant()

            # interpolate squared function
            interpolant_squared = KernelInterpolant(kernel, dim)
            interpolant_squared.generate_design_points(Ndesign, method=type_design_points)
            # sample and build
            xdesign = interpolant_squared.xdesign
            interpolant_squared.ydesign = test_fct(xdesign)**2
            interpolant_squared.build_interpolant()

            # compute datapoint
            data[idx_dim, test_fct_idx] = interpolant_squared.native_norm()/interpolant.native_norm()**2
                
    exp_data['data'] = data
    return exp_data


def run_feminterpolation_experiment(exp_data):
    # setup from exp_data
    kernel_class = exp_data.get('kernel_class')
    dim_list = exp_data.get('dim_list')
    weights_decay = exp_data.get('weights_decay')
    num_design_shifts   = exp_data.get('num_design_shifts')
    num_error_shifts   = exp_data.get('num_error_shifts')
    type_design_points = exp_data.get('type_design_points', 'lattice')
    Ndesign_list = exp_data.get('Ndesign_list')
    meshsize = exp_data.get('meshsize', 10)
    if isinstance(weights_decay, Number):
        weights_decay = (weights_decay, weights_decay)
    elif type(weights_decay) not in [tuple, list, np.array]:
        TypeError('weights_decay has to be a number or a tuple/list/np.array of two numbers')
    error_samples = 2**7

    # interpolation
    errors = np.empty((len(dim_list), len(Ndesign_list)))
    for idx_dim, dim in enumerate(dim_list):
        elliptic_bvp_fe = EllipticProblemFE(dim, meshsize, q=weights_decay[0])
        gamma = [1/j**weights_decay[1] for j in range(1,dim+1)]
        kernel = kernel_class(dim, lengthscales=gamma)
        for idx_Ndesign, Ndesign in enumerate(Ndesign_list):
            interpolants = interpolate_multivar(elliptic_bvp_fe, meshsize, kernel, Ndesign, num_design_shifts, type_design_points)
            errors[idx_dim, idx_Ndesign] = \
                np.sqrt(np.mean([L2errorFEM(interpolants[design_shift,:], error_samples, meshsize, elliptic_bvp_fe.f, elliptic_bvp_fe.q, qmc_shifts=num_error_shifts) for design_shift in range(num_design_shifts)]))
    exp_data['error_data'] = errors
    return exp_data


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
    experiment_object = experiment_function(experiment_object)
    duration = time.time() - t1
    print(f"Total runtime: {duration//(60*60*24):.0f}d {duration//(60*60)%24:2.0f}h {duration//(60)%60:2.0f}m {duration%60:2.0f}s")

    write_experiment_data(experiment_object)