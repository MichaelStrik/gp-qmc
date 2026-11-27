import itertools
import time
from os import cpu_count
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Number

import numpy as np
import qmcpy
from scipy.special import zeta
from multiprocess import Process, Queue

from kernels import UnanchoredSobolevKernel, AnchoredSobolevKernel
from interpolant import KernelInterpolant
from testproblems import TestFunction, EllipticProblemQOI, EllipticProblemFE
from utils import worker, get_touchard_polynomial


class AbstractExperiment(ABC):
    """
    Abstract class implementing the procedure of high-dimensional kernel interpolation experiment.

    The experiment configuration is obtained from experiment_data. The core functionality is run_experiment which 
    for every dimensionality in experiment_data["dim_list"] and every number of design points in experiment_data["Ndesign_list"]
    constructs a design and an interpolant for a (possibly multivariate) target for which then error data is computed. Several
    error data can be computed for one interpolant, which is to be specified by error_length.

    Attributes:
        experiment_data : dict
            Experiment configuration data like, e.g., a list of numbers of design points and a number of error samples to use.
        kernel_class : class
            A class implementing a kernel.
        target_class : class
            A callable class implementing a target function to be interpolated.
    """
    def __init__(self, experiment_data=None, error_length=1):
        self.experiment_data = experiment_data
        self.error_length = error_length
        self.errors       = None

    @abstractmethod
    def instantiate_kernel(self, input_dim):
        """Instantiates the kernel for a given dimension dim. This method can be overriden for adaption to the specific interface of kernel_class."""
        pass
    
    @abstractmethod
    def instantiate_target(self, input_dim, output_dim):
        """Instantiates the target for a given dimension dim. This method can be overriden for adaption to the specific interface of target_class."""
        pass

    @abstractmethod
    def determine_output_dim(self, input_dim):
        """ Returns an output dimension based on input_dim. """
        pass

    @abstractmethod
    def compute_error(self, interpolants, input_dim):
        """ Computes an error of length error_length for each interpolant in interpolants. """
        pass

    @abstractmethod
    def construct_design(self, n, dim):
        """ 
        The implementation should return an indexable collection of design points. The first index should be reserved for differentiation across different sets of design points. 
        The second index selects a design points, while the the third index selects a design point's component. The sets of design points do not necessarily have to be of the same length.
        """
        pass

    def construct_interpolant(self, xdesigns, input_dim):
        output_dim = self.determine_output_dim(input_dim)
        num_designs = len(xdesigns)
        interpolants = np.empty((num_designs, output_dim), dtype=KernelInterpolant)

        # decide on parallelism
        if output_dim*num_designs > 0.7*cpu_count():
            # process-level parallelism
            N_PROCESSES = cpu_count()
            threadlimit = 1
        else:
            # thread-level parallelism
            N_PROCESSES = 1
            threadlimit = cpu_count()
        task_queue = Queue()
        done_queue = Queue()
        def evaluate_target(target, kernel, xdesign, idx_design_set, **kwargs):
            ydesign = target(xdesign, **kwargs)
            return ydesign, idx_design_set
        def interpolate(target, kernel, xdesign, ydesign, idx_design_set, idx_outdim, **kwargs):
            interpolant = KernelInterpolant(kernel, kernel.d, xdesign, ydesign)
            interpolant.build_interpolant(**kwargs)
            return interpolant, idx_design_set, idx_outdim
        processes = []
        TASKS = [(evaluate_target, (xdesign, idx_design_set)) for (idx_design_set, xdesign) in enumerate(xdesigns)]
        for task in TASKS:
            task_queue.put(task)
        for n in range(N_PROCESSES):
            processes.append(Process(target=self.parallel_worker_interpolate, args=(task_queue, done_queue, input_dim, threadlimit)))
            processes[-1].start()
        for task in TASKS:
            ydesign, idx_design_set = done_queue.get()
            for idx_outdim in range(output_dim):
                task_queue.put((interpolate, (xdesigns[idx_design_set], ydesign[:,idx_outdim], idx_design_set, idx_outdim)))
        for idx_design_set, idx_outdim in itertools.product(range(num_designs), range(output_dim)):
            interpolant, idx_design_set, idx_outdim = done_queue.get()
            interpolants[idx_design_set, idx_outdim] = interpolant
        for process in processes:
            task_queue.put('STOP')
        for process in processes:
            process.join()
        return interpolants

    def parallel_worker_interpolate(self, input, output, input_dim, threadlimit):
        target = self.instantiate_target(input_dim, self.determine_output_dim(input_dim))
        kernel = self.instantiate_kernel(input_dim)
        kwargs = {'threadlimit':threadlimit}
        for func, args in iter(input.get, 'STOP'):
            args = (target, kernel) + args
            result = func(*args, **kwargs)
            output.put(result)

    def run_experiment(self):
        # preparation
        exp_data            = self.experiment_data
        dim_list            = exp_data.get('dim_list')
        Ndesign_list = exp_data.get('Ndesign_list')
        self.errors = np.empty((len(dim_list), len(Ndesign_list), self.error_length))
        # experiment logic
        for input_dim in dim_list:
            print(f"\tdim {input_dim}")
            for n in Ndesign_list:
                print(f"\t\tn {n}")
                design = self.construct_design(n, input_dim)
                timestamp=time.time()
                interpolants = self.construct_interpolant(design, input_dim)
                print(f"\t\tBuilt interpolants after {(time.time() - timestamp):.1f}s.")
                timestamp=time.time()
                self.errors[input_dim, n] = self.compute_error(interpolants, input_dim)
                print(f"\t\tSampled errors after {(time.time() - timestamp):.1f}s.")

    def obtain_result(self):
        """
        Copies errors into experiment_data and returns the latter.
        
        This function may be overridden for the users requirements when deriving a subclass.
        """
        self.experiment_data["error_data"] = self.errors
        return self.experiment_data



class Experiment_FEMWeightedNorm(AbstractExperiment):
    r"""
    Implements an experiment for the interpolation of finite element solutions of the PDE-class EllipticProblemFE.

    Two error criteria are computed: \sqrt{\int_{[0,1]^s} ||u_sh - u_shn||^2} where || \cdot || is the weighted L2 or energy norm with weight function psi.

    Attributes:
        stochastic_transform: A string specifying the stochastic transform used for the coefficient function. This information also determines its type.
    """
    def __init__(self, experiment_data):
        super().__init__(experiment_data, error_length=2)
        self.type_design_points = experiment_data.get("type_design_points")
        self.num_design_shifts = experiment_data.get("num_design_shifts")
        # weights
        weights_decay = experiment_data.get("weights_decay")
        if isinstance(weights_decay, Number):
            weights_decay = (weights_decay, weights_decay)
        elif type(weights_decay) not in [tuple, list, np.ndarray]:
            raise TypeError(  
                'Type of weights_decay needs to be one of [tuple, list, np.ndarray]. Hint: It should contain a single number containing ' \
                'the decay or a pair of two numbers, the first specifying the decay of weights in the problem and analogously the second ' \
                'one in the method.'
                )
        self.product_weights_decay_target       = weights_decay[0]
        self.product_weights_decay_interpolant  = weights_decay[1]
        # target
        self.target_class = EllipticProblemFE
        self.stochastic_transform = experiment_data.get("stochastic_transform")
        self.meshsize = experiment_data.get("meshsize")
        # interpolants
        if type(experiment_data["kernel_class"]) is str:
            kernel_dict = {"UnanchoredSobolevKernel":UnanchoredSobolevKernel, "AnchoredSobolevKernel":AnchoredSobolevKernel}
            kernel_name = experiment_data["kernel_class"]
            experiment_data["kernel_class"] = kernel_dict[kernel_name]
        self.kernel_class = experiment_data.get("kernel_class")
        # error
        self.num_design_shifts  = experiment_data.get("num_design_shifts")
        self.num_error_shifts   = experiment_data.get("num_error_shifts")

    def instantiate_kernel(self, input_dim):
        """Returns a kernel instance for a given dimension dim. This method can be overriden for adaption to the specific interface of kernel_class."""
        # extract weights
        # pass over weight to constructor of kernel class
        q = self.product_weights_decay_interpolant
        prod_weights = self.prod_weights(input_dim, q)
        return self.kernel_class(input_dim, weights_dimension=prod_weights)

    def instantiate_target(self, input_dim, output_dim):
        """Returns a target instance for a given dimension dim. This method can be overriden for adaption to the specific interface of target_class."""
        return EllipticProblemFE(input_dim, self.meshsize, self.stochastic_transform, f=lambda x: x[0], q=self.product_weights_decay_target)

    def determine_output_dim(self, input_dim):
        """Returns the output dimension of the target."""
        return self.meshsize

    @staticmethod
    def b(j, c, q):
        a_min = 1-c/np.sqrt(6)*zeta(q)
        return c*j**(-q)/(np.sqrt(6)*a_min)
    
    def prod_weights(self, dim, q):
        c = EllipticProblemFE.c
        p = 1/2.2
        delta = 1e-6
        sigma = 1
        llambda = 1/(2/p - 1 - 2*sigma - 4*delta)
        touchard_poly = get_touchard_polynomial(sigma)
        return [(((j*sigma)**sigma*touchard_poly(Experiment_FEMWeightedNorm.b(j, c, q)))**2/(2*np.exp(1/np.e)*zeta(2*llambda)))**(1/(1+llambda)) for j in range(1,dim+1)]

    def construct_design(self, n : int, dim : int):
        """
        Returns a collection of self.num_design_shifts many sets of design points of the type self.type_design_points.
        
        The type self.type_design_points can be one of 'mc' (Monte Carlo), 'lattice' (QMC/lattice) and 'digital_net' (QMC/digital net).
        The name self.num_design_shifts suggests that the code is used with lattices, but but several sets of design points could also 
        be generated with digital nets and Monte Carlo points with different meanings (see qmcpy documentation for digital nets). 
        Args:
            n   : Number of points to construct. For lattices and digital nets, n must be a power of 2.
            dim : Dimension of the points constructed.
        Returns:
            design : Array of design points. The first index differentiates sets of design points, the second index points and the third index components.
        """
        type_design_points = self.type_design_points.lower()
        if type_design_points not in ['lattice', 'digital_net', 'mc']:
            raise TypeError("Unknown type of design points. self.type_design_points must be one of ['lattice', 'digital_net', 'mc'].")
        # lattice
        if type_design_points == 'lattice':
            lattice = qmcpy.Lattice(dimension=dim, replications=self.num_design_shifts)
            design = lattice(n)
        # digital net
        elif type_design_points == 'digital_net':
            digital_net = qmcpy.DigitalNetB2(dimension=dim, replications=self.num_design_shifts)
            design = digital_net(n)
        # monte carlo
        elif type_design_points == 'mc':
            rng = np.random.default_rng()
            design = rng.uniform(0, 1, size=(self.num_design_shifts, n, dim))
        return design

    @staticmethod
    def psi(y, alpha):
        """
        Product weight function psi(y) = prod_j exp(-alpha_j * |y_j|) = exp(-sum_j alpha_j * |y_j|). Intended for a weighted function norm on the stochastic domain.

        Args:
            y     : array of shape (n, s), row i is a point in R^s
            alpha : array of shape (s,), with 0 < alpha_min <= alpha_j <= alpha_max
        Returns: array of shape (n,)
        """
        y = np.asarray(y, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        return np.exp(-np.abs(y) @ alpha)

    def compute_error(self, interpolants, input_dim):
        """Computes an error of length error_length for each interpolant in interpolants.
        
        References:
            [1] Graham, I.G., Kuo, F.Y., Nichols, J.A., Scheichl, R., Schwab, Ch., Sloan, I.H.: Quasi-Monte Carlo finite element methods for 
                elliptic PDEs with lognormal random coefficients. Numer. Math. 131, 329-368 (2015)
        """
        # generate qmc points sets. one for each error shift
        num_design_shifts = self.num_design_shifts
        num_error_shifts = self.num_error_shifts
        num_error_samples = 2**int(self.experiment_data.get("error_samples"))
        errsample_point_sets = qmcpy.Lattice(dimension=input_dim, replications=self.num_error_shifts)(num_error_samples, warn=False)
        # choose alpha for weighted norm according to [1, Chapter 4]
        c = EllipticProblemFE.c
        q = self.product_weights_decay_target
        b = np.array([c*j**(-q) for j in range(1, input_dim+1)])
        alpha = np.maximum(b, 0.01*np.ones_like(b))
        # prepare parallel computing
        processes = []
        task_queue = Queue()
        done_queue = Queue()
        def qmc_worker(input, output):
            target = self.instantiate_target(input_dim, self.determine_output_dim(input_dim))
            for interpolants, errsample_point_subset in iter(input.get, 'STOP'):
                # evaluate interpolants to obtain degree of freedom vectors
                dofs_interp = np.empty((len(errsample_point_subset), len(interpolants)))
                for idx, interpolant in enumerate(interpolants):
                    dofs_interp[:,idx] = interpolant(errsample_point_subset)
                sum_L2_err_local = 0
                sum_energy_err_local = 0
                # compute errors to FEM solutions
                for idx, point in enumerate(errsample_point_subset):
                    L2_err, energy_err = target.error_to_fem(point, dofs_interp[idx,:],threadlimit=1)
                    sum_L2_err_local += self.psi(point, alpha)*L2_err
                    sum_energy_err_local += self.psi(point, alpha)*energy_err
                output.put(sum_L2_err_local, sum_energy_err_local)
        N_PROCESSES = cpu_count()
        for i in range(N_PROCESSES):
            processes.append(Process(target=qmc_worker, args = (task_queue, done_queue)))
        # allocate
        Qsa_L2      = np.zeros(num_design_shifts) # shift-average QMC rules
        Qsa_energy  = np.zeros(num_design_shifts)
        Q_L2     = np.zeros(num_error_shifts)     # QMC rules
        Q_energy = np.zeros(num_error_shifts)
        L2_local_sums = np.zeros(N_PROCESSES)     # a worker's contribution to a QMC rule
        energy_local_sums = np.zeros(N_PROCESSES) # is stored here
        # compute
        for des_shift in range(num_design_shifts):
            for err_shift in range(num_error_shifts):
                # apply qmc rule
                errsample_point_set = errsample_point_sets[err_shift]
                errsample_point_subsets = np.array_split(errsample_point_set, N_PROCESSES)
                for i in range(N_PROCESSES):
                    task_queue.put((interpolants[des_shift,:], errsample_point_subsets[i]))
                for i in range(N_PROCESSES):
                    L2_local_sums[i], energy_local_sums[i] = done_queue.get()
                # compute shift-average QMC rule
                Q_L2[err_shift] = np.sum(L2_local_sums)/num_error_samples
                Q_energy[err_shift] = np.sum(energy_local_sums)/num_error_samples
            Qsa_L2[des_shift]     = np.mean(Q_L2)
            Qsa_energy[des_shift] = np.mean(Q_energy)
        # compute design-shift-averages of the QMC-estimated error norms
        L2_desshift_avg     = np.mean(Qsa_L2)
        energy_desshift_avg = np.mean(Qsa_energy)
        return L2_desshift_avg, energy_desshift_avg


def run_feminterpolation_experiment_new(exp_data):
    experiment = Experiment_FEMWeightedNorm(exp_data)
    experiment.run_experiment()
    return experiment.obtain_result()


def interpolate(test_function, kernel, Ndesign, num_design_shifts, type_design_points='lattice'):
    dim = kernel.d
    if type(test_function) is TestFunction:
        # for the artificial problem, the kernel norm is cheap to compute
        test_function_norm = test_function.kernel_norm()
    else:
        # otherwise, we leave normalisation to the error computing function
        test_function_norm = None

    # parameters for error approximation
    k = 10
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
        # print("\t\tBuilding interpolant")
        interpolant.build_interpolant()
        # print("\t\tSampling error function")
        errors[shift] = interpolant.error_L2squared(test_function, Nsamples=2**k, qmc_shifts=num_error_shifts, normalisation=test_function_norm) # TODO Is normalisation by the squared norm more appropriate?
        interpolants.append(interpolant)

    return interpolants, errors


def interpolate_multivar(test_function, output_dim, kernel, Ndesign, num_design_shifts, type_design_points='lattice'):
    timestamp = time.time()
    print("\t\tBuilding interpolants.")
    input_dim = kernel.d
    interpolants = np.empty((num_design_shifts, output_dim), dtype=KernelInterpolant)
    xdesigns = np.empty((num_design_shifts, Ndesign, input_dim))
    ydesigns = np.empty((num_design_shifts, Ndesign, output_dim))
    for shift in range(num_design_shifts):
        interpolant = KernelInterpolant(kernel, input_dim) # this interpolant is only needed to generate one common design
        interpolant.generate_design_points(Ndesign, method=type_design_points)
        xdesigns[shift] = interpolant.xdesign
        ydesigns[shift] = test_function(interpolant.xdesign)
    # build interpolants in parallel
    if output_dim*num_design_shifts > 0.7*cpu_count():
        # process-level parallelism
        N_PROCESSES = cpu_count()
        threadlimit = 1
    else:
        # thread-level parallelism
        N_PROCESSES = 1
        threadlimit = cpu_count()
    task_queue = Queue()
    done_queue = Queue()
    def generate_interpolant(interpolant, xdesign, ydesign, shift, outdim):
        interpolant.xdesign = xdesign
        interpolant.ydesign = ydesign
        interpolant.build_interpolant(threadlimit)
        return interpolant, shift, outdim
    TASKS = [(generate_interpolant, (KernelInterpolant(kernel, input_dim), xdesigns[shift], ydesigns[shift,:,outdim], shift, outdim)) for (shift, outdim) in itertools.product(range(num_design_shifts), range(output_dim))]
    for task in TASKS:
        task_queue.put(task)
    processes = []
    for n in range(N_PROCESSES):
        processes.append(Process(target=worker, args=(task_queue, done_queue)))
        processes[-1].start()
    for task in TASKS:
        interpolant, shift, outdim = done_queue.get()
        interpolants[shift, outdim] = interpolant
    for n in range(N_PROCESSES):
        task_queue.put('STOP')
    for process in processes:
        process.join()
    # for shift in range(num_design_shifts):
    #     for outdim in range(output_dim):
    #         interpolant = KernelInterpolant(kernel, input_dim)
    #         interpolant.xdesign = xdesign
    #         interpolant.ydesign = ydesign[:,outdim]
    #         interpolant.build_interpolant()
    #         interpolants[shift, outdim] = interpolant
    print(f"\t\tBuilt interpolants with {(time.time() - timestamp):.1f}s runtime.")
    return interpolants


def errorsFEM(interpolants: np.array, Nsamples, Nx, transform, f:Callable, q, method='lattice', qmc_shifts=1, **kwargs):
    r"""
    Computes the L2-L2 error 
        $$ \int_U \int_D (u_h(x,y) - u_{h,n}(x,y))^2 dx dy $$

    and the L2-energy error 
        $$ \int_U \int_D (\nabla u_h(x,y) - \nabla u_{h,n}(x,y))^2 dx dy $$ 
    
    over the physical domain D \subset \R and the stochastic domain U for FEM solutions u_h and kernel interpolants u_{h,n}.

    Args:
        interpolants:np.array
        Collection of interpolants that together represent a FEM interpolation. 
        A fixed mesh is assumed and there should be one interpolant for each degree of freedom.
        
        Nsamples : int
        Number of samples.

        Nx : int 
        Degrees of freedom of finite element method.

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
    Q_L2_shifted = np.zeros(qmc_shifts)
    Q_energy_shifted = np.zeros(qmc_shifts)

    def worker_error_to_fem(input, output, ydim, Nx, transform, f, q):
            elliptic_problem = EllipticProblemFE(ydim, Nx, transform, f, q)
            for pts, coefs in iter(input.get, 'STOP'):
                assert(pts.shape[0] == coefs.shape[0])
                Q_L2_local = 0
                Q_energy_local = 0
                for y, coef_interp in zip(pts,coefs):
                    L2error, energy_error = elliptic_problem.error_to_fem(y, coef_interp)
                    Q_L2_local += L2error
                    Q_energy_local += energy_error
                output.put((Q_L2_local, Q_energy_local))
    N_PROCESSES = cpu_count()-1
    task_queue = Queue()
    done_queue = Queue()
    for shift in range(qmc_shifts):
        points = point_sets[shift,:,:]

        # calculate interpolated fem coefficients at points
        fem_coefficients = np.empty((points.shape[0],Nx))
        # timestamp = time.time()
        def evaluate(i, interpolant, points):
            coefs = interpolant(points)
            return i, coefs
        TASKS = [(evaluate, (i, interpolant, points)) for i, interpolant in enumerate(interpolants)]
        for task in TASKS:
            task_queue.put(task)
        processes = []
        for n in range(N_PROCESSES):
            processes.append(Process(target=worker, args=(task_queue, done_queue)))
            processes[-1].start()
        for task in TASKS:
            i, coefs = done_queue.get()
            fem_coefficients[:,i] = coefs
        for n in range(N_PROCESSES):
            task_queue.put('STOP')
        for process in processes:
            process.join()

        # for i, interpolant in enumerate(interpolants):
        #     fem_coefficients[:,i] = interpolant(points)
        # duration = time.time() - timestamp
        # print(f"\t\t\tInterpolating coefficients took {duration:.2f}s.")

        # at sampling points, calculate L2 distance between surrogate and FEM approximations
        points_splits = np.array_split(points, N_PROCESSES)
        fem_coefficients_splits = np.array_split(fem_coefficients, N_PROCESSES)
        TASKS = [(pts,coefs) for (pts,coefs) in zip(points_splits,fem_coefficients_splits)]
        for task in TASKS:
            task_queue.put(task)
        processes = []
        for n in range(N_PROCESSES):
            processes.append(Process(target=worker_error_to_fem, args=(task_queue, done_queue, ydim, Nx, transform, f, q)))
            processes[-1].start()
        for task in TASKS:
            Q_L2_local, Q_energy_local = done_queue.get()
            Q_L2_shifted[shift] += Q_L2_local
            Q_energy_shifted[shift] += Q_energy_local
        for n in range(N_PROCESSES):
            task_queue.put('STOP')
        for process in processes:
            process.join()
        Q_L2_shifted[shift] /= Nsamples
        Q_energy_shifted[shift] /= Nsamples
    return np.mean(Q_L2_shifted), np.mean(Q_energy_shifted)


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
    elif type(weights_decay) not in [tuple, list, np.ndarray]:
        TypeError('weights_decay has to be a number or a tuple/list/np.array of two numbers')

    # experiment
    for idx_dim, dim in enumerate(dim_list):
        print(f"\tDimension {dim}")
        gamma = [1/j**weights_decay[1] for j in range(1,dim+1)]
        kernel = kernel_class(dim, weights_dimension=gamma)

        elliptic_bvp = EllipticProblemQOI(dim, q=weights_decay[0])
        for idx_Ndesign, Ndesign in enumerate(Ndesign_list):
            print(f"\t\tNdesign {Ndesign}")
            t1 = time.time()
            interpolants_design_shifts, errors_design_shifts = interpolate(elliptic_bvp, kernel, Ndesign, num_design_shifts, type_design_points)
            errors[idx_dim, idx_Ndesign]            = np.sqrt(np.mean(errors_design_shifts)) # shift-average error
            condition_numbers[idx_dim, idx_Ndesign] = np.mean([interpolants_design_shifts[j].condition_number for j in range(num_design_shifts)]) # shift-average condition number
            duration = time.time() - t1
            print(f"\t\tFinished with runtime {duration:.2f}")
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
        kernel = kernel_class(dim, weights_dimension=gamma)
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
            kernel = kernel_class(dim, weights_dimension=gamma)

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
    
    data = np.zeros((len(dim_list), num_test_functions))
    
    for idx_dim, dim in enumerate(dim_list):
        print(f"dim {dim}")
        # kernel
        gamma = [1/j**weights_decay for j in range(1,dim+1)]
        kernel = kernel_class(dim, weights_dimension=gamma)
        
        # construct a number of random RKHS functions
        test_functions_list = []
        for i in range(num_test_functions):
            test_functions_list.append(TestFunction(kernel, dim, num_anchors=dim*10))

        for test_fct_idx, test_fct in enumerate(test_functions_list):
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
    def b(j, c, q):
        a_min = 1-c/np.sqrt(6)*zeta(q)
        return c*j**(-q)/(np.sqrt(6)*a_min)
    def prod_weights(dim, q):
        c = 0.2
        p = 1/2.2
        delta = 1e-6
        sigma = 1
        llambda = 1/(2/p - 1 - 2*sigma - 4*delta)
        touchard_poly = get_touchard_polynomial(sigma)
        return [(((j*sigma)**sigma*touchard_poly(b(j, c, q)))**2/(2*np.exp(1/np.e)*zeta(2*llambda)))**(1/(1+llambda)) for j in range(1,dim+1)]
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
    elif type(weights_decay) not in [tuple, list, np.ndarray]:
        TypeError('weights_decay has to be a number or a tuple/list/np.array of two numbers')
    error_samples = 2**10
    stochastic_transform_name = exp_data.get('stochastic_transform', 'uniform').lower()

    # interpolation
    errors = np.empty((len(dim_list), len(Ndesign_list), 2)) # last axis is to distinguish L2 and energy errors
    for idx_dim, dim in enumerate(dim_list):
        print(f"\tDimension {dim}")
        elliptic_bvp_fe = EllipticProblemFE(dim, meshsize, stochastic_transform_name, f=lambda x: x[0], q=weights_decay[0])
        gamma = prod_weights(dim, weights_decay[1])
        kernel = kernel_class(dim, weights_dimension=gamma)
        for idx_Ndesign, Ndesign in enumerate(Ndesign_list):
            print(f"\t\tNdesign {Ndesign}")
            interpolants = interpolate_multivar(elliptic_bvp_fe, meshsize, kernel, Ndesign, num_design_shifts, type_design_points)
            timestamp = time.time()
            errors[idx_dim, idx_Ndesign, :] = \
                np.sqrt(np.mean([errorsFEM(interpolants[design_shift,:], error_samples, meshsize, stochastic_transform_name, elliptic_bvp_fe.f, elliptic_bvp_fe.q, qmc_shifts=num_error_shifts) for design_shift in range(num_design_shifts)], axis=0))
            print(f"\t\tFinished error sampling with {(time.time() - timestamp):.1f}s runtime.")
    exp_data['error_data'] = errors
    return exp_data


def write_experiment_data(experiment):
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