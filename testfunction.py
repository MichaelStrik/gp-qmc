import numpy as np


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
            One column corresponds to one anchor.

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

        self.anchors = rng.uniform(0, 1, size=(dim, num_anchors))
        self.coefficients = rng.normal(size=num_anchors)


    def kernel_norm(self):
        norm_squared = 0
        coefficients = self.coefficients
        anchors = self.anchors
        kernel = self.kernel
        for i in range(self.num_anchors):
            for j in range(self.num_anchors):
                norm_squared += coefficients[i]*coefficients[j]*kernel(anchors[:,i], anchors[:,j])

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
            values[:,i] = self.kernel(x, anchors[:,i])

        return np.dot(values, coefficients)



if __name__ == '__main__':
    # visuals
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.tri import Triangulation
    # kernel
    from kernels import UnanchoredSobolevKernel
    def buildTestFunction(num_anchors=10):
        kernel = UnanchoredSobolevKernel(2, lengthscales=[1, 0.5])
        return TestFunction(kernel, 2, num_anchors)

    # build test function
    print("Testing: TestFunction.")
    print("Building kernel-based test function.")
    test_function = buildTestFunction(num_anchors=2)
    print("Done.")

    # visualise
    nx, ny = (101, 101)
    X = np.linspace(-0.5, 0.5, nx)
    Y = np.linspace(-0.5, 0.5, ny)
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