# qmcpy
from qmcpy.kernel.si_dsi_kernels import AbstractSIDSIKernel
from qmcpy.util.shift_invar_ops import bernoulli_poly
from qmcpy.util.transforms import tf_exp_eps,tf_exp_eps_inv


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
        if self.npt.any(beta0) or self.npt.any(beta1):
            raise NotImplementedError("Given beta0 or beta1 not None. Derivatives are not implemented.")
        
        delta = self.npt.abs(x0-x1)
        kperdim = self.npt.stack([self.npt.concatenate([1/2*bernoulli_poly(2, delta[...,j,None]) + (x0[...,j,None]-1/2)*(x1[...,j,None]-1/2) for j in range(self.d)],-1)],-2)
        return kperdim



class AnchoredSobolevKernel(AbstractSIDSIKernel):
    """Represents the reproducing kernel of the weighted unanchored Sobolev space.

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
    
    
    def eta(self, x0, x1):
        npt = self.npt
        c = self.anchor
        out = self.npt.zeros_like(x0)
        both_greater = npt.all([x0 > c, x1 > c], axis=0) # all corresponds to boolean and, any to boolean or
        both_smaller = npt.all([x0 < c, x1 < c], axis=0)
        out[both_greater] = self.npt.maximum(x0,x1) - c
        out[both_smaller] = c - self.npt.minimum(x0,x1)
        out[not npt.any([both_greater, both_smaller], axis=0)] = 0
        
        return out


    def get_per_dim_components(self, x0, x1, beta0, beta1):
        if self.npt.any(beta0) or self.npt.any(beta1):
            raise NotImplementedError("Given beta0 or beta1 not None. Derivatives are not implemented.")
        
        kperdim = self.npt.stack([self.npt.concatenate([  self.eta(x0[...,j,None],x1[...,j,None]) for j in range(self.d)],-1)],-2)
        return kperdim




if __name__ == '__main__':
    import numpy as np

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
    
    # scribble
    x0 = np.array(([0.0, 0.5], [0.5, 0.5]))
    x1 = np.array(([0.0, 0.5], [0.0, 0.0]))
    gamma = [1.0, 2.0]
    UnancSobolevKernel = UnanchoredSobolevKernel(2, lengthscales=gamma)
    kmat = UnancSobolevKernel(x0, x1)


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
