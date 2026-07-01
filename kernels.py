# qmcpy
from qmcpy.kernel.si_dsi_kernels import AbstractSIDSIKernel
from qmcpy.util.shift_invar_ops import bernoulli_poly
from qmcpy.util.transforms import tf_exp_eps,tf_exp_eps_inv
import numpy as np


class AbstractWeightedKernel:
    """
    Abstract class for kernels with product weights or with product and order dependent (POD) weights. 
    For the product case, one only needs to specify weights for each individual dimension (dimension_weights),
    whereas for the POD case, there should also be specified weights for every order (cardinality of a set of dimensions).
    The type of weights will be inferred automatically from he datatype of order_weights.
    """
    def __init__(self,
                d,
                scale = 1.,
                dimension_weights = 1.,
                order_weights = 1.
                ):
        self.d = d
        self.scale = scale
        self.dimension_weights = dimension_weights
        self.order_weights = order_weights
        self.P_table = None


    def eta(self,x,y):
        raise NotImplementedError


    # PRODUCT WEIGHTS
    def get_per_dim_components(self, x0, x1):
        kperdim = np.stack([self.eta(x0[...,j], x1[...,j]) for j in range(self.d)],-1)
        return kperdim
    

    def combine_per_dim_components(self, kperdim):
        scale = self.scale
        weights = self.dimension_weights
        k = scale * ((1+weights*kperdim).prod(-1))
        return k
    

    # POD WEIGHTS
    def P(self,x0,x1,s,l):
        if l > s:
            return 0
        elif l == 0:
            return 1
        else:
            if not np.any(np.isnan(self.P_table[s,l,...])):
                return self.P_table[s,l,...]
            P = self.P(x0,x1,s-1,l) + self.dimension_weights[s-1]*self.eta(x0[...,s-1],x1[...,s-1])*self.P(x0,x1,s-1,l-1)
            self.P_table[s,l,...] = P
            return P
        
    
    def sum_P(self,x0,x1):
        d = self.d
        order_weights = self.order_weights
        # allocate P_table
        if x0.ndim==3 and x1.ndim==3:
            n = np.max(x0.shape[:-1])
            m = np.max(x1.shape[:-1])
            # we need to predict the output shape
            # the last shape entry is the dimensionality of the input points/of the kernel and doesn't matter
            # if ndim==3, we assume shapes of the form (n,1,d) or (1,n,d) for a broadcasted calculation, with n>=1 being the number of points
            self.P_table = np.nan*np.ones((d+1,d+1,n,m))
        elif x0.ndim==2 and x1.ndim==2:
            # in this case, x0 and x1 are matrices where two rows of the same index form a pair of points, therefore they should have the same amount of rows
            n = x0.shape[0]
            m = x1.shape[0]
            assert n == m
            self.P_table = np.nan*np.ones((d+1,d+1,n))
        elif x0.ndim==1 and x1.ndim==1:
            self.P_table = np.nan*np.ones((d+1,d+1))
        else:
            raise TypeError("Shapes not compatible.")
        
        k = 0
        scale = self.scale
        for l in range(d+1):
            k += scale*order_weights[l]*self.P(x0,x1,d,l)
        self.P_table = None

        return k


    def __call__(self, x0, x1):
        # assertions
        # TODO

        if np.isscalar(self.order_weights):
            # product weights
            kperdim = self.get_per_dim_components(x0,x1)
            k = self.combine_per_dim_components(kperdim)
            return k
        else:
            # product and order dependent (POD) weights
            return self.sum_P(x0,x1)


class UnanchoredSobolevKernel(AbstractSIDSIKernel):
    """Represents the reproducing kernel of the weighted unanchored Sobolev space.

    Can evaluate the kernel for a given pair of points, dimension, weights, etc.
    The evaluation is vectorised, in a call kernel(x, y) where x, y are matrices
    and kernel is an instance of this class, every row of x or y is 
    interpreted as one vector in IR^d and the kernel evaluates per row, that is one row constitutes one pair of points.
    The only exception is x having a multiple rows and y a single one (or vice versa), then the kernel evaluates for every
    row of x with the single row in y.
    If evaluation for every possible pair of rows in x and rows in y is desired, we can use a numpy indexing trick 
    and access according to kernel(x[:,np.newaxis,:], y[np.newaxis,:,:]) before calling the kernel. 
    Here, x and y can have different numbers of rows, but need to have the same numbers of columns (dimensions) of course.
    This is best understood by looking at the simpler operation x[:,np.newaxis,:] + y[np.newaxis,:,:].
    Useful resources:
        https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools
        https://numpy.org/doc/stable/user/basics.broadcasting.html
    """
    def __init__(self,
            d, 
            scale = 1., 
            lengthscales = 1.,
            alpha = 2,
            shape_scale = [1],
            shape_lengthscales = None, 
            tfs_scale = (tf_exp_eps_inv,tf_exp_eps),
            tfs_lengthscales = (tf_exp_eps_inv,tf_exp_eps),
            torchify = False, 
            requires_grad_scale = True, 
            requires_grad_lengthscales = True, 
            device = "cpu",
            compile_call = False,
            comiple_call_kwargs = {},
            ):
        r"""
        Args:
            d (int): Dimension. 
            scale (Union[np.ndarray,torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray,torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray,torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
            shape_scale (list): Shape of `scale` when `np.isscalar(scale)`. 
            shape_lengthscales (list): Shape of `lengthscales` when `np.isscalar(lengthscales)`
            tfs_scale (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_lengthscales (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            torchify (bool): If `True`, use the `torch` backend. Set to `True` if computing gradients with respect to inputs and/or hyperparameters.
            requires_grad_scale (bool): If `True` and `torchify`, set `requires_grad=True` for `scale`.
            requires_grad_lengthscales (bool): If `True` and `torchify`, set `requires_grad=True` for `lengthscales`.
            device (torch.device): If `torchify`, put things onto this device.
            compile_call (bool): If `True`, `torch.compile` the `parsed___call__` method. 
            comiple_call_kwargs (dict): When `compile_call` is `True`, pass these keyword arguments to `torch.compile`.
        """
        super().__init__(
            d = d, 
            scale = scale, 
            lengthscales = lengthscales,
            alpha = alpha, 
            shape_scale = shape_scale,
            shape_lengthscales = shape_lengthscales, 
            tfs_scale = tfs_scale, 
            tfs_lengthscales = tfs_lengthscales, 
            torchify = torchify,
            requires_grad_scale = requires_grad_scale,
            requires_grad_lengthscales = requires_grad_lengthscales,
            device = device,
            compile_call = compile_call,
            comiple_call_kwargs = comiple_call_kwargs,
        )
    
    
    def get_per_dim_components(self, x0, x1, beta0, beta1):
        npt = self.npt
        if npt.any(beta0) or npt.any(beta1):
            raise NotImplementedError("Given beta0 or beta1 not None. Derivatives are not implemented.")
        
        delta = npt.abs(x0-x1)
        kperdim = npt.stack([npt.concatenate([1/2*bernoulli_poly(2, delta[...,j,None]) + (x0[...,j,None]-1/2)*(x1[...,j,None]-1/2) for j in range(self.d)],-1)],-2)
        return kperdim



class AnchoredSobolevKernel(AbstractSIDSIKernel):
    """Represents the reproducing kernel of the weighted anchored Sobolev space.

    Can evaluate the kernel for a given pair of points, dimension, weights, etc.
    The evaluation is vectorised, in a call kernel(x, y) where x, y are matrices
    and kernel is an instance of this class, every row of x respectively y is 
    interpreted as one vector in IR^d.
    """
    def __init__(self,
            d, 
            anchor = 0,
            scale = 1., 
            lengthscales = 1.,
            alpha = 2,
            shape_scale = [1],
            shape_lengthscales = None, 
            tfs_scale = (tf_exp_eps_inv,tf_exp_eps),
            tfs_lengthscales = (tf_exp_eps_inv,tf_exp_eps),
            torchify = False, 
            requires_grad_scale = True, 
            requires_grad_lengthscales = True, 
            device = "cpu",
            compile_call = False,
            comiple_call_kwargs = {},
            ):
        r"""
        Args:
            d (int): Dimension. 
            anchor (float): anchor of the anchored kernel between 0 and 1
            scale (Union[np.ndarray,torch.Tensor]): Scaling factor $S$.
            lengthscales (Union[np.ndarray,torch.Tensor]): Product weights $(\gamma_1,\dots,\gamma_d)$.
            alpha (Union[np.ndarray,torch.Tensor]): Smoothness parameters $(\alpha_1,\dots,\alpha_d)$ where $\alpha_j \geq 1$ for $j=1,\dots,d$.
            shape_scale (list): Shape of `scale` when `np.isscalar(scale)`. 
            shape_lengthscales (list): Shape of `lengthscales` when `np.isscalar(lengthscales)`
            tfs_scale (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_lengthscales (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            torchify (bool): If `True`, use the `torch` backend. Set to `True` if computing gradients with respect to inputs and/or hyperparameters.
            requires_grad_scale (bool): If `True` and `torchify`, set `requires_grad=True` for `scale`.
            requires_grad_lengthscales (bool): If `True` and `torchify`, set `requires_grad=True` for `lengthscales`.
            device (torch.device): If `torchify`, put things onto this device.
            compile_call (bool): If `True`, `torch.compile` the `parsed___call__` method. 
            comiple_call_kwargs (dict): When `compile_call` is `True`, pass these keyword arguments to `torch.compile`.
        """
        super().__init__(
            d = d, 
            scale = scale, 
            lengthscales = lengthscales,
            alpha = alpha, 
            shape_scale = shape_scale,
            shape_lengthscales = shape_lengthscales, 
            tfs_scale = tfs_scale, 
            tfs_lengthscales = tfs_lengthscales, 
            torchify = torchify,
            requires_grad_scale = requires_grad_scale,
            requires_grad_lengthscales = requires_grad_lengthscales,
            device = device,
            compile_call = compile_call,
            comiple_call_kwargs = comiple_call_kwargs,
        )
        self.anchor = anchor
    
    
    def compute_eta(self, x0, x1):
        """
        Computes eta as in [1, Chapter 4.2] per dimension. 
        
        Args:
            x0: np.array
            Column vector or scalar
            
            x1: np.array
            Column vector or scalar

        [1] J. Dick, F. Kuo and I. Sloan: High-dimensional integration: The quasi-Monte Carlo way. In: Acta Numerica 22 (2013), pp. 133-288
        """
        npt = self.npt
        c = self.anchor
        eta = npt.zeros_like(x0+x1) # have to use the shape that results after broadcasting

        both_greater = npt.logical_and(x0 > c, x1 > c)
        both_smaller = npt.logical_and(x0 < c, x1 < c)

        eta = npt.where(both_greater, npt.minimum(x0,x1) - c, eta)
        eta = npt.where(both_smaller, c - npt.maximum(x0,x1), eta)

        return eta


    def eta2(self, x0, x1):
        """
        Computes eta as in [1, Chapter 4.2] per dimension. 
        
        Args:
            x0: np.array
            Column vector or scalar
            
            x1: np.array
            Column vector or scalar

        [1] J. Dick, F. Kuo and I. Sloan: High-dimensional integration: The quasi-Monte Carlo way. In: Acta Numerica 22 (2013), pp. 133-288
        """
        npt = self.npt
        c = self.anchor
        out = npt.zeros_like(x0+x1) # have to use the shape that results after broadcasting

        both_greater = npt.logical_and(x0 > c, x1 > c)
        both_smaller = npt.logical_and(x0 < c, x1 < c)

        # we distinguish between cases because broadcasting interferes the indexing
        if x0.shape == x1.shape:
            out[both_greater] =     npt.minimum(x0[both_greater],x1[both_greater]) - c
            out[both_smaller] = c - npt.maximum(x0[both_smaller],x1[both_smaller])
        elif x0.shape != out.shape: # x0 gets broadcast
            # we can and do make assumptions about the shapes here. 
            # eta is computed per dimension, so x0 and x1 have shape (num_pts,1) or (1,) (maybe also (1,1)). 
            # broadcasting only happens when a shape entry is one, so in that case we can assume having shape (1,) for the broadcast array
            out[both_greater] =     npt.minimum(x0[npt.any(both_greater), None], x1[both_greater]) - c
            out[both_smaller] = c - npt.maximum(x0[npt.any(both_smaller), None], x1[both_smaller])
        elif x1.shape != out.shape: # x1 gets broadcast
            out[both_greater] =     npt.minimum(x0[both_greater], x1[[npt.any(both_greater)]]) - c
            out[both_smaller] = c - npt.maximum(x0[both_smaller], x1[[npt.any(both_smaller)]])
        
        out[~npt.logical_or(both_greater, both_smaller)] = 0
        
        return out


    def get_per_dim_components(self, x0, x1, beta0, beta1):
        if self.npt.any(beta0) or self.npt.any(beta1):
            raise NotImplementedError("Given beta0 or beta1 not None. Derivatives are not implemented.")
        
        kperdim = self.npt.stack([self.npt.concatenate([  self.compute_eta(x0[..., j, None], x1[..., j, None]) for j in range(self.d)],-1)],-2)
        return kperdim



if __name__ == '__main__':
    import numpy as np

    dim = 10
    UnancSobolevKernel = UnanchoredSobolevKernel(dim, lengthscales=[1/j**2 for j in range(1,dim+1)])
    rng = np.random.default_rng()
    x_array = rng.uniform(0, 1, size=(3, dim))
    kmat = UnancSobolevKernel(x_array, x_array)


    dim = 1
    AncSobolevKernel = AnchoredSobolevKernel(dim, anchor=0, lengthscales=[1/j**2 for j in range(1,dim+1)])
    k = AncSobolevKernel(np.array([0.001]),np.array([0.4]))

    dim = 10
    AncSobolevKernel = AnchoredSobolevKernel(dim, anchor=0, lengthscales=[1/j**2 for j in range(1,dim+1)])
    rng = np.random.default_rng()
    x_array = rng.uniform(0, 1, size=(3, dim))
    kvec = AncSobolevKernel(x_array, x_array[0,:])


    # test against more manual calculations
    def manualCalc_unanc(x0, x1, gamma=[1.0, 1.0]):
        # evaluation of unanchored sobolev kernel, d=2 and weights as given
        gamma1 = gamma[0]
        gamma2 = gamma[1]
        d1 = x0[None,0]-x1[None,0]
        d2 = x0[None,1]-x1[None,1]
        eta1 = 1/2*bernoulli_poly(2, np.abs(d1)) + (x0[0]-1/2)*(x1[0]-1/2)
        eta2 = 1/2*bernoulli_poly(2, np.abs(d2)) + (x0[1]-1/2)*(x1[1]-1/2)

        k = (1+gamma1*eta1)*(1+gamma2*eta2)
        return k[0]

    def manualCalc_anc(x0, x1, gamma=[1.0, 1.0]):
        # evaluation of anchored sobolev kernel with anchor c=0, dimension d=2 and weights as given
        return (1+gamma[0]*np.minimum(x0[0], x1[0]))*(1+gamma[1]*np.minimum(x0[1], x1[1]))

    x0 = np.array([0.0, 0.5])
    x1 = np.array([0.0, 0.5])
    gamma = [1.0, 2.0]

    UnancSobolevKernel = UnanchoredSobolevKernel(2, lengthscales=gamma)
    kmat = UnancSobolevKernel(x0, x1)
    print("Testing: UnanchoredSobolevKernel.")
    print(f"Kernel class.       kmat: {kmat}")
    print(f"Manual calculation. kmat: {manualCalc_unanc(x0, x1, gamma)}")
    print("-----\n")


    AncSobolevKernel = AnchoredSobolevKernel(2, anchor=0, lengthscales=gamma)
    print("Testing: AnchoredSobolevKernel.")
    print(f"Error: {AncSobolevKernel(x0, x1) - manualCalc_anc(x0, x1, gamma)}")

    print("----\nKernel reimplementation")
    def test(kernel):
        x = np.array([0.0,2.0])
        y = np.array([3.0,4.0])
        z = np.array([6.0,9.0])
        print(f"k(x,y)={kernel(x,y)}")
        print(f"k(y,z)={kernel(y,z)}")
        print(f"k(x,z)={kernel(x,z)}")
        print(f"k(x,x)={kernel(x,x)}")
        print(f"k(y,y)={kernel(y,y)}")
        print(f"k(z,z)={kernel(z,z)}")
        S = np.stack([x,y,z])
        print(kernel(S[:,np.newaxis,:], S[np.newaxis,:,:]))
        # print(S[:,np.newaxis,:].shape, S[np.newaxis,:,:].shape)
        # print(S[:,np.newaxis,:].ndim)

    kernelOld = UnanchoredSobolevKernel(d=2, lengthscales=[1,2])
    test(kernelOld)

    print("\n----"+"\nNew weighted kernel")
    class PODWeightsUnanchoredSobolevKernel(AbstractWeightedKernel):
        def eta(self,x0,x1):
            # one-dimensional for now, without broadcasting
            return 1/2*(np.abs(x0-x1)**2 -np.abs(x0-x1) +1/6) +(x0-1/2)*(x1-1/2)

    kernel = PODWeightsUnanchoredSobolevKernel(d=2, dimension_weights=[1,2])
    test(kernel)

    # print("Step by step")
    # x0 = np.array([0.0,0.0])
    # x1 = np.array([1.0,1.0])
    # kperdim = kernel.get_per_dim_components(x0,x1)
    # kperdimOld = kernelOld.get_per_dim_components(x0,x1,[],[])
    # print(f"kperdim = {kperdim}, kperdimOld = {kperdimOld}")
    # weights = kernel.dimension_weights
    # print(f"Manually: {(1+weights*kperdim).prod(-1)}")
    # print(f"Kernel: {kernel(x0,x1)}")
    # kernelOld(x0,x1)

    # def etaOld(x,y):
    #     delta = np.array(np.abs(x-y))
    #     return 1/2*bernoulli_poly(2, delta) + (x-1/2)*(y-1/2)
    # print(f"Etas: {etaOld(0,1),kernel.eta(0,1)}")
    # bernoulli_poly(2,np.array(3.0))