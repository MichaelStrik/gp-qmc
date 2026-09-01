import numpy as np
import qmcpy
from multiprocess import Process, Queue
from os import cpu_count

from numbers import Number
from utils import worker, print_runtime

from threadpoolctl import threadpool_limits


class KernelInterpolant:
    """Represents a kernel interpolant on the domain [0, 1]^d.


    Attributes:
        kernel : function
        kernel(x1 : np.array, x2 : np.array)
        Takes two input vectors x1, x2 and returns a scalar value. This is the heart of the interpolation method, 
        the kernel completely determines the characteristics of the interpolant.

        dim : int
        Dimensionality. The kernel must represent a mathematical function k: IR^dim x IR^dim --> IR.

        xdesign : np.array
        Design points. These are the points in space at which we can assume we know the value of our target function.
        In practice we usually assume the target function to be expensive in evaluation, but nonetheless available, 
        so that we simply evaluate it at the selected design points.

        ydesign : np.array
        Function data at design points. These are the values of our target function at the design points xdesign.

        alpha : np.array
        Coefficients for the design points determined by a linear system. Changing the coefficients means altering the interpolant.

    Methods:
        generate_design_points

        build_interpolant

        __call__
    """

    def __init__(self, kernel, dim, xdesign=None, ydesign=None):
        self.kernel  = kernel
        self.dim     = dim
        self.xdesign = xdesign
        self.ydesign = ydesign
        
        self.condition_number = None
        self.alpha = None


    def generate_design_points(self, Ndesign, method='lattice', **kwargs):
        """Chooses design points according to the given method.

        The user can choose between lattice rules and digital nets (QMC) and more simple Monte Carlo (MC) points.

        Args:
            Ndesign : int
            Number of points to construct. Note the restrictions from the methods, often Ndesign has to be a power of 2.

            method : str
            Can be one of 'lattice' and 'digital_net'.

            **kwargs : dict
            Collects all keyword-arguments that aren't parameters of this function and passes them over to the chosen QMC method.
        """
        method = method.lower()
        assert method in ['lattice', 'digital_net', 'mc']

        if method == 'lattice':
            lattice = qmcpy.Lattice(dimension=self.dim, **kwargs)
            self.xdesign = lattice(Ndesign)

        elif method == 'digital_net':
            digital_net = qmcpy.DigitalNetB2(dimension=self.dim, **kwargs)
            self.xdesign = digital_net(Ndesign)

        elif method == 'mc':
            rng = np.random.default_rng()
            self.xdesign = rng.uniform(0, 1, size=(Ndesign, self.dim))


    def build_interpolant(self):
        """Builds the kernel based interpolant.
        """ 
        # print(f"Condition number: {self.condition_number:.2e}")
        K = self.kernel(self.xdesign[:,np.newaxis,:], self.xdesign[np.newaxis,:,:])
        with threadpool_limits(limits=cpu_count()-1):
            self.condition_number = np.linalg.cond(K)
            alpha = np.linalg.solve(K,self.ydesign)
    
        self.alpha = alpha


    def __call__(self, xeval):
        """Evaluates the kernel based interpolant at the points given in xeval.
        """
        if self.alpha is None:
            raise RuntimeError( "The interpolant was called, but self.alpha is None. " + \
                                "Please run self.build_interpolant() before calling the interpolant.")

        if len(xeval.shape) == 1:
            if self.dim==1:
                xeval = xeval[:,np.newaxis]
            else:
                xeval = xeval[np.newaxis,:]
        # evaluate interpolant at points xeval
        # if shape[0]==1:
        #     Keval = self.kernel(xeval, self.xdesign)
        #     yeval = np.dot(Keval,(self.alpha))

        Keval = self.kernel(xeval[:,np.newaxis,:], self.xdesign[np.newaxis,:,:])
        # with threadpool_limits(limits=cpu_count()-1):
        yeval = Keval@(self.alpha)

        return yeval
    

    def native_norm(self):
        """
        Computes native/reproducing kernel Hilbert space norm of the interpolant.
        """
        alpha = self.alpha
        X = self.xdesign
        K = self.kernel(X[:,np.newaxis,:], X[np.newaxis,:,:])

        return np.inner(alpha@K,alpha)


    def error_L2squared(self, target, Nsamples, method='lattice', qmc_shifts=1, normalisation=None, **kwargs):
        """Returns an approximation to the L2-error of the interpolant w.r.t. target.
        Args:
            target : function(x)
            The interpolation target. Must represent a mathematical function IR^dim --> IR.

            Nsamples : int
            Number of samples.

            method : str
            Method. One of 'lattice', 'digital_net', 'MC' (Monte Carlo).
        """
        method = method.lower()
        assert method in ['mc', 'lattice', 'digital_net']
        
        if method == 'mc':
            rng = np.random.default_rng()
            point_sets = rng.uniform(0, 1, size=(Nsamples, self.dim))
            point_sets = point_sets[np.newaxis,:,:]
            qmc_shifts = 1
            # target_eval = target(point_set)
            # d = self(point_set) - target_eval
            # abserror_squared = np.sum(d**2)/Nsamples
            # L2norm_squared = np.sum(target_eval**2)/Nsamples
        # qmc
        elif method == 'lattice' or method == 'digital_net':
            qmcGens = {'lattice': qmcpy.Lattice, 'digital_net': qmcpy.DigitalNetB2}
            qmcGen = qmcGens[method](dimension=self.dim, replications=qmc_shifts, **kwargs)
            point_sets = qmcGen(Nsamples, warn=False)
        
        # shift-average qmc estimator
        Q_s = np.zeros(qmc_shifts)
        L2norm_squared = 0
        def calc_d(pts):
            target_eval = target(pts)
            self_eval = self(pts)
            d = target_eval - self_eval
            return np.sum(d**2), np.sum(target_eval**2)
        N_PROCESSES = 1#cpu_count()
        task_queue = Queue()
        done_queue = Queue()
        for r in range(qmc_shifts):
            points = point_sets[r,:,:]
            points_subviews = np.array_split(points, N_PROCESSES)
            TASKS = [(calc_d, (pts,)) for pts in points_subviews]
            for task in TASKS:
                task_queue.put(task)
            for i in range(N_PROCESSES):
                Process(target=worker, args=(task_queue, done_queue)).start()
            for i in range(len(TASKS)):
                d_squared, target_eval_squared = done_queue.get()
                Q_s[r] += d_squared
                L2norm_squared += target_eval_squared
            for i in range(N_PROCESSES):
                task_queue.put('STOP')
            Q_s[r] /= Nsamples
            L2norm_squared /= Nsamples
        abserror_squared = np.mean(Q_s)
        
        if type(normalisation) is str and normalisation.upper() == 'L2':
            normalisation = L2norm_squared
        elif not isinstance(normalisation, Number):
            normalisation = 1

        return abserror_squared/normalisation



if __name__ == '__main__':
    print("Running randomised interpolation test case.")
    from kernels import UnanchoredSobolevKernel, AnchoredSobolevKernel
    from testproblems import TestFunction

    # build kernel
    d = 25
    Ndesign = 256
    weights_decay = 2.5
    unanc_sob_kernel = UnanchoredSobolevKernel(d, lengthscales=[1/j**weights_decay for j in range(1,d+1)])
    anc_sob_kernel = AnchoredSobolevKernel(d, lengthscales=[1/j**weights_decay for j in range(1,d+1)])

    # construct qmc points
    print("Constructing QMC points.")
    interpolant = KernelInterpolant(anc_sob_kernel, d)
    
    interpolant.generate_design_points(Ndesign)

    # build test function and provide data
    print("Building test function.")
    test_function = TestFunction(unanc_sob_kernel, d, num_anchors=10)
    xdesign = interpolant.xdesign
    interpolant.ydesign = test_function(xdesign)

    # build interpolant and approximate error
    print("Building interpolant.")
    interpolant.build_interpolant()
    k = 11
    qmc_shifts = 5
    print(f"Error L2: {interpolant.error_L2squared(test_function, Nsamples=2**k, qmc_shifts=qmc_shifts)}")
    
