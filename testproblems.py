import numpy as np
from scipy.integrate import solve_bvp, quad
# parallel
from multiprocess import Process, Queue
from os import cpu_count
from utils import worker


class TestFunction():
    """Represents a relatively simple function in the RKHS of a given kernel intended as a benchmarking/test problem.
    
    The function maps IR^dim --> IR and is constructed via a linear combination of anchored kernels, i.e. kernels where the first argument 
    is fixed and in the second argument the input of the test function is inserted. In that sense, one anchor constitutes a function with 
    one input which we also refer to as an anchored kernel function. Thus, the resulting test function is then member of the reproducing-kernel 
    Hilbert space (RKHS) associated to the kernel.
    The anchor points and coefficients are randomly chosen by default, but can be manipulated by the user after initialisation of the object.

    Attributes:

        kernel : function
        Kernel function, takes two arguments from IR^dim and outputs a real number.

        dim : int
        Input dimension of the test function.

        anchors : np.array, size=(dim, num_anchors)
        Array of kernel anchors. One column corresponds to one anchor, a vector in IR^dim.

        num_anhors : int
        Number of anchors and thus number of anchored kernel functions in the linear combination.

        coefficients : np.array
        Coefficients to the num_anchors many anchored kernel functions.



    Methods:

        randomise(dim, num_anchors)
        Chooses 'num_anchors' many anchor points in IR^dim and 'num_anchors' many coefficients. 
        Anchor points are drawn from a uniform distribution Unif([-1/2, 1/2]^dim), while coefficients are drawn from a normal N(0, diag(1))^dim.

    """

    def __init__(self, kernel, dim, num_anchors):
        """

        Args:
            kernel : function(x,y)
            Kernel function. When applying kernel interpolation to the test function, the kernel used in the anchored kernel functions and the kernel used 
            in the kernel interpolation method should be the same for the test function to be in the RKHS.

            dim : int
            Dimensionality of the function

            anchors : np.array
            One row corresponds to one anchor.

            num_anchors : int
            Number of anchors and hence of anchored kernel functions.
        
        Methods:
            randomise

            kernel_norm
            Returns the kernel-induced norm of the test function.

            __call__
            Evaluates the test function.
        """
        
        self.kernel = kernel
        self.dim = dim
        self.num_anchors = num_anchors

        self.randomise(dim, num_anchors)


    def randomise(self, dim, num_anchors):
        """Chooses random kernel anchor points and coefficients.
        """
        rng = np.random.default_rng()

        self.anchors = rng.uniform(0, 1, size=(num_anchors, dim))
        self.coefficients = rng.normal(size=num_anchors)


    def kernel_norm(self):
        norm_squared = 0
        coefficients = self.coefficients
        anchors = self.anchors
        kernel = self.kernel
        for i in range(self.num_anchors):
            for j in range(self.num_anchors):
                norm_squared += coefficients[i]*coefficients[j]*kernel(anchors[i,:], anchors[j,:])

        return np.sqrt(norm_squared)


    def __call__(self, x):
        """Returns the test function's value at x.

        Args:
            x : np.array
            Input vector. If x is a matrix, each of its rows is interpreted as a single vector.
        """
        anchors      = self.anchors
        num_anchors  = self.num_anchors
        coefficients = self.coefficients

        if x.ndim == 2:
            num_inputs, _ = x.shape
        else:
            num_inputs = 1

        values = np.zeros((num_inputs, num_anchors))
        for i in range(num_anchors):
            values[:,i] = self.kernel(x, anchors[i,:])

        return np.dot(values, coefficients)



class EllipticProblemQOI():
    def __init__(self, dim, f = lambda x: np.zeros_like(x), q=4/3):
        self.dim = dim
        self.f = f
        self.q = q


    # ode solver
    def a(self, x, y):
        q = self.q
        pi = np.pi
        
        result = np.ones_like(x)
        for j in range(self.dim):
            result += y[j]/(1+(j*pi)**q)*np.sin(j*pi*x)
        return result


    def dx_a(self, x, y):
        q = self.q
        pi = np.pi
        
        result = np.zeros_like(x)
        for j in range(self.dim):
            result += y[j]*j*pi/(1+(j*pi)**q)*np.cos(j*pi*x)
        return result


    def solve(self, y, Nx = 2, tol=1e-5):
        """
        y:  parameter
        Nx: initial mesh resolution
        """
        def rhs(x, u):
            u0 = u[[1]]
            u1 = (self.f(x) - self.dx_a(x,y))/self.a(x,y)
            return np.vstack((u0, u1))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])
            
        # mesh and initial guess
        x = np.linspace(0, 1, num=Nx)
        u = np.zeros((2,Nx))

        return solve_bvp(rhs, bc, x, u, tol=tol)
    
    
    # call
    def __call__(self, y):
        if y.ndim == 1:
            y = y[np.newaxis,:]
            y_rows = 1
            y_cols = y.shape[1]
        elif y.ndim==2:
            y_rows, y_cols = y.shape
        assert y_cols == self.dim
        
        qoi = np.zeros(y_rows) # quantity of interest

        # parallelisation helpers
        def worker(input, output):
            for func, args in iter(input.get, 'STOP'):
                result = func(*args)
                output.put(result)
        def calc_qoi(idx, y_row):
            solobj = self.solve(y_row)
            u_sol = lambda x: solobj.sol(x)[0]
            res = quad(u_sol, a=1/8, b=3/8)[0]
            return idx, res
        N_PROCESSES = cpu_count()
        task_queue = Queue()
        done_queue = Queue()
        TASKS = [(calc_qoi, (row_idx, y_row)) for row_idx, y_row in enumerate(y)]
        for task in TASKS:
            task_queue.put(task)
        for i in range(N_PROCESSES):
            Process(target=worker, args=(task_queue, done_queue)).start()
        for task in TASKS:
            idx, res = done_queue.get()
            qoi[idx] = res
        for i in range(N_PROCESSES):
            task_queue.put('STOP')

        # for row in range(y_rows):
        #     solobj = self.solve(y[row,:])
        #     u_sol = lambda x: solobj.sol(x)[0]
        #     qoi[row] = quad(u_sol, a=1/8, b=3/8)[0]

        return qoi


class EllipticProblemFE():
    def __init__(self, ydim, Nx, f = lambda x: np.zeros_like(x), q=4/3):
        from mpi4py import MPI
        from dolfinx import mesh, fem, default_scalar_type
        import ufl

        # spaces
        domain = mesh.create_unit_interval(MPI.COMM_SELF, Nx-1)
        V = fem.functionspace(domain, ("Lagrange", 1))

        # boundary conditions
        uD = fem.Constant(domain, default_scalar_type(0.0))
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(uD, boundary_dofs, V)


        self.dim = ydim
        self.f = f
        self.q = q

        self.domain = domain
        self.V = V
        self.bc = bc
        self.Nx = Nx


    # fem solver
    def c_unif(self, x, y):
        import ufl
        q = 2
        pi = np.pi
        dim = self.dim

        result = 1
        for j in range(dim):
            result += y[j]/(1+(j*pi)**q)*ufl.sin(j*pi*x[0])

        return result
    

    def solve(self, y):
        from dolfinx import fem, default_scalar_type
        from dolfinx.fem.petsc import LinearProblem
        import ufl
        V = self.V
        domain = self.domain
        bc = self.bc

        # variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, default_scalar_type(0.0))
        
        x = ufl.SpatialCoordinate(domain)
        a = self.c_unif(x, y)*ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        # assemble & solve
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="Poisson",
        )
        problem.solve()

        return problem

    
    def L2error_to_fem(self, y, Nx, coefficients):
        from mpi4py import MPI
        from dolfinx import fem
        import ufl
        u_h     = self.solve(y).u
        u_hs    = fem.Function(self.V, coefficients) # surrogate/interpolant
        L2error = fem.form(ufl.inner(u_h-u_hs,u_h-u_hs)*ufl.dx)
        error_local = fem.assemble_scalar(L2error)
        L2error = np.sqrt(self.domain.comm.allreduce(error_local, op=MPI.SUM))

        return L2error


    def __call__(self, y):
        if y.ndim == 1:
            y = y[np.newaxis,:]
            y_rows = 1
            y_cols = y.shape[1]
        elif y.ndim==2:
            y_rows, y_cols = y.shape
        assert y_cols == self.dim
        
        fem_coefficients = np.zeros((y_rows, self.Nx))

        def pde_worker(input, output, ydim, Nx, f, q):
            elliptic_problem = EllipticProblemFE(ydim, Nx, f, q)
            for row_idx, y in iter(input.get, 'STOP'):
                sol = elliptic_problem.solve(y)
                coefficients = np.array(sol.x)
                output.put((row_idx, coefficients))
        N_PROCESSES = cpu_count()
        task_queue = Queue()
        done_queue = Queue()
        TASKS = [(row_idx, y_row) for row_idx, y_row in enumerate(y)]
        for task in TASKS:
            task_queue.put(task)
        for n in range(N_PROCESSES):
            Process(target=pde_worker, args=(task_queue, done_queue, self.dim, self.Nx, self.f, self.q)).start()
        for task in TASKS:
            row_idx, coefs = done_queue.get()
            fem_coefficients[row_idx,:] = coefs
        for n in range(N_PROCESSES):
            task_queue.put('STOP')

        return fem_coefficients


if __name__ == '__main__':
    # visuals
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    # kernel
    from kernels import UnanchoredSobolevKernel, AnchoredSobolevKernel

    def buildTestFunction(kernel_class, dim=2, lengthscales=[1, 0.5], num_anchors=10):
        kernel = kernel_class(dim, lengthscales=lengthscales)
        return TestFunction(kernel, dim, num_anchors)

    dim = 2
    lengthscales=[1, 0.5]
    
    # build test function with unanchored kernel
    print(f"Building test function based on {UnanchoredSobolevKernel.__name__}.")
    test_function = buildTestFunction(UnanchoredSobolevKernel, dim, lengthscales, num_anchors=2)
    print("Done.")


    # build test function with unanchored kernel
    print(f"Building test function based on {AnchoredSobolevKernel.__name__}.")
    test_function = buildTestFunction(AnchoredSobolevKernel, dim, lengthscales, num_anchors=2)
    print("Done.")


    # visualise
    nx, ny = (101, 101)
    X = np.linspace(0, 1, nx)
    Y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(X, Y)
    xv = xv.flatten() # put the coordinates into a long, thin matrix
    yv = yv.flatten()
    points = np.stack((xv, yv), axis=-1)

    Z = test_function(points)

    # surface plot
    triang = Triangulation(xv, yv) # there is some shape issue here probably. haven't read https://stackoverflow.com/questions/55577981/plotly-plot-trisurf-isnt-working-with-arange-arrays yet
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(xv, yv, Z, linewidth=0.2, antialiased=True)
    ax.set_title("3D Surface Plot: Randomly generated, kernel-based function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("sin(x*y)")

    plt.show()


    # 1d test
    kernel_class = AnchoredSobolevKernel
    print(f"Building test function based on {kernel_class.__name__}.")
    test_function = buildTestFunction(kernel_class, dim=1, lengthscales=[1], num_anchors=2)
    
    X = np.linspace(0,1, 1000)
    Y = test_function(X[:,None])
    ax = plt.plot(X, Y)
    plt.title(f"Test function based on {kernel_class.__name__}")
    plt.show()