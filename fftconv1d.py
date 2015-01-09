import os
import theano
import theano.tensor as T
import theano.sandbox.cuda.fftconv as fftci
from theano.sandbox.cuda.fftconv import cufft, cuifft, batched_complex_dot
import numpy as np
from theano import Apply
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.opt import (
    local_optimizer, gpu_optimizer, gpu_seqopt)
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous, HostFromGpu)
from theano.sandbox.cuda import basic_ops
from theano.ifelse import ifelse

def mult_and_reduce(input_fft_v, filters_fft_v, input_shape=None,
                    filter_shape=None):
    """
    input_fft_v is (b, ic, i1//2 + 1, 2)
    filters_fft_v is (oc, ic, i1//2 + 1, 2)
    """

    if input_shape is None:
        input_shape = input_fft_v.shape  # symbolic

    if filter_shape is None:
        filter_shape = filters_fft_v.shape  # symbolic

    b, ic, i1_f, _ = input_shape
    oc = filter_shape[0]

    # reshape to flatten the dimensions that are multiplied elemwise
    input_r = input_fft_v.reshape((b, ic, i1_f, 2))
    filters_r = filters_fft_v.reshape((oc, ic, i1_f, 2))

    # shuffle for batched dot product
    input_s = input_r.dimshuffle(2, 0, 1, 3)  # (i1_f, b, ic, 2)
    filters_s = filters_r.dimshuffle(2, 1, 0, 3)  # (i1_f, ic, oc, 2)

    output_s = batched_complex_dot(input_s, filters_s)

    # shuffle again
    output_r = output_s.dimshuffle(1, 2, 0, 3)

    # reshape to unflatten
    output = output_r.reshape((b, oc, i1_f, 2))

    return output

def core_conv1d_fft(input, filters, border_mode='valid', image_shape = None, filter_shape = None, pad_last_dim=True):
    """
    Perform a convolution through fft.

    Only support input which will be even on the last dimension
    (width).  All other dimensions can be anything and the filters can
    have an even or odd width.

    If you must use input which has an odd width, you can either pad
    it or use the `pad_last_dim` argument which will do it for you and
    take care to strip the padding before returning.  Don't use this
    argument if you are not sure the input is odd since the padding is
    unconditional and will make even input odd, thus leading to
    problems.

    On valid mode the filters must be smaller than the input.

    input: (b, ic, i0)
    filters: (oc, ic, f0)

    border_mode: 'valid' of 'full'

    pad_last_dim: Unconditionally pad the last dimension of the input
                  to to turn it from odd to even.  Will strip the
                  padding before returning the result.
    """
    
    # use symbolic shapes to compute shape info at runtime if not specified
    if image_shape is None:
        image_shape = input.shape

    if filter_shape is None:
        filter_shape = filters.shape

    # batch size, input channels, input dim 0
    b, ic, i0 = image_shape
    # output channels, input channels, filter dim 0
    oc, ic_, f0 = filter_shape

    # pad filters/image to output shape
    if border_mode == 'valid':
        if pad_last_dim:
            is_odd = T.eq(T.mod(i0, 2), 1)
            o0 = ifelse(is_odd, i0 + 1, i0)
            input_padded = T.zeros((b, ic, o0), dtype='float32')
            input_padded = T.set_subtensor(input_padded[:, :, :i0],
                                       input)
        else:
            o0 = i0
            input_padded = input

        filters_padded = T.zeros((oc, ic, o0), dtype='float32')
        filters_padded = T.set_subtensor(filters_padded[:, :, :f0],
                                         filters)

    elif border_mode == 'full':

        # In this particular case, the values of (o0, o1) represent
        # the dimensions of the work buffer more than the actual dimensions
        # of the desired output.
        o0 = i0 + 2 * (f0 - 1)

        if pad_last_dim:
            is_odd = T.eq(T.mod(o0, 2), 1)
            o0 = ifelse(is_odd, o0 + 1, o0)

        # We line up the filters and the images in a way
        # such that the filters are tightly placed against the
        # top-left of the array, and the images intersect with
        # them on one pixel. The top-left pixel of the images
        # is the bottom-right pixel of the filters when we
        # do the layout here.

        filters_padded = T.zeros((oc, ic, o0), dtype='float32')
        filters_padded = T.set_subtensor(filters_padded[:, :, :f0],
                                         filters)

        input_padded = T.zeros((b, ic, o0), dtype='float32')
        input_padded = T.set_subtensor(input_padded[:, :, (f0 - 1):(f0 - 1 + i0)],
                                       input)
    else:
        raise ValueError('invalid mode')

    input_padded = T.opt.Assert("in conv1d_fft: width is not even")(
        input_padded, T.eq(o0 % 2, 0))

    # reshape for FFT
    input_flat = input_padded.reshape((b * ic, o0))
    filters_flat = filters_padded.reshape((oc * ic, o0))

    # perform FFT
    input_fft_flat = cufft(input_flat)  # (b * ic, o0, o1//2 + 1, 2)
    filters_fft_flat = cufft(filters_flat)  # (oc * ic, o0, o1//2 + 1, 2)

    # unfold ic dimension
    input_fft_v_shape = (b, ic, o0 // 2 + 1, 2)
    filters_fft_v_shape = (oc, ic, o0 // 2 + 1, 2)
    input_fft_v = input_fft_flat.reshape(input_fft_v_shape)
    filters_fft_v = filters_fft_flat.reshape(filters_fft_v_shape)

    # (b, oc, o0, o1//2 + 1, 2)
    output_fft_s = mult_and_reduce(input_fft_v, filters_fft_v,
                                   input_shape=input_fft_v_shape,
                                   filter_shape=filters_fft_v_shape)

    # reshape for IFFT
    output_fft_flat = output_fft_s.reshape((b * oc, o0 // 2 + 1, 2))

    # perform IFFT
    output_flat = cuifft(output_fft_flat)  # (b * oc, o0, o1)

    # reshape
    output_circ = output_flat.reshape((b, oc, o0))  # circular!

    # Now we extract the region of interest.
    # We just cut it out from the output_circ
    # array that was used for the computation.
    # We do not need to handle pad_last_dim in a
    # special way because we specify explicitly here
    # how much values are expas_cuda_ndarray_variableected.
    if border_mode == 'valid':
        output = output_circ[:, :, (f0-1):(f0-1 + i0-f0+1)]
    elif border_mode == 'full':
        output = output_circ[:, :, (f0-1):(f0-1 + i0+f0-1)]
    else:
        raise ValueError('invalid mode')

    # Rescale manually. This is just a factor that comes in during the
    # trip through FFT and inverse FFT.
    output = (1.0 / T.cast(o0, 'float32')) * output

    # output should now be the result of a batched valid convolution
    # of the input with the filters.
    ret = basic_ops.as_cuda_ndarray_variable(output)
    ret = T.patternbroadcast(ret,(input.type.broadcastable[0],
                         filters.type.broadcastable[0],
                         False))
    return ret

from theano.sandbox.cuda import GpuOp
from theano.tensor import TensorType
class Symbol1DConvOp(GpuOp):

    def make_node(self, img, kern):
        if img.type.ndim != 3:
            raise TypeError('img must be 3D tensor')
        if kern.type.ndim != 3:
            raise TypeError('kern must be 3D tensor')

        broadcastable = (img.type.broadcastable[0],
                         kern.type.broadcastable[0],
                         False)
        #print img.type,kern.type
        return Apply(self, [img, kern],
                     [CudaNdarrayType(broadcastable)()])

    def perform(self,*args,**kwargs):
        try:
            print args,kwargs
            raise Exception()
        except:
            import traceback
            traceback.print_exc()
            raise
class Symbol1DConvGradI(Symbol1DConvOp):
    def __init__(self, border_mode, image_shape=None,filter_shape=None): self.border_mode, self.image_shape, self.filter_shape = border_mode, image_shape, filter_shape
class Symbol1DConvGradW(Symbol1DConvOp):
    def __init__(self, border_mode, image_shape=None,filter_shape=None): self.border_mode, self.image_shape, self.filter_shape = border_mode, image_shape, filter_shape
class Symbol1DConv(Symbol1DConvOp):
    def __init__(self, border_mode='valid', image_shape=None,filter_shape=None):
        self.border_mode, self.image_shape, self.filter_shape = border_mode, image_shape, filter_shape
        if image_shape!=None and filter_shape!=None:
            if border_mode == 'valid':
                self.output_shape = [image_shape[0], filter_shape[0], image_shape[2]-filter_shape[2]+1]
            else:
                self.output_shape = [image_shape[0], filter_shape[0], image_shape[2]+filter_shape[2]-1]
        else:
            self.output_shape = None

    def grad(self,inp,grads):
        img, kerns = inp
        top, = grads
        d_img = Symbol1DConvGradI(self.border_mode, self.filter_shape, self.output_shape)(kerns, top)
        d_kerns = Symbol1DConvGradW(self.border_mode, self.image_shape, self.output_shape)(img, top)
        
        return d_img, d_kerns

from theano.sandbox.cuda.opt import register_opt
@register_opt('fft1d')
@local_optimizer([Symbol1DConv],True)
def local_s1dconv(node):
    if isinstance(node.op, Symbol1DConv):
        img, kern = node.inputs
        return [core_conv1d_fft(img, kern, node.op.border_mode)]
@register_opt('fft1d')
@local_optimizer([Symbol1DConvGradI])
def local_s1dconvgI(node):
    if isinstance(node.op, Symbol1DConvGradI):
        kerns, top = node.inputs
        def T(s):
            if s==None: return None
            return [s[i] for i in [1,0,2]]
        return [core_conv1d_fft(top, kerns.dimshuffle(1,0,2)[:,:,::-1], 'full' if node.op.border_mode=='valid' else 'valid', node.op.filter_shape, T(node.op.image_shape))]
@register_opt('fft1d')
@local_optimizer([Symbol1DConvGradW])
def local_s1dconvgW(node):
    if isinstance(node.op, Symbol1DConvGradW):
        img, top = node.inputs
        def T(s):
            if s==None: return None
            return [s[i] for i in [1,0,2]]
        if node.op.border_mode=='valid':
            r = [basic_ops.as_cuda_ndarray_variable(core_conv1d_fft(img.dimshuffle(1,0,2)[:,:,::-1], top.dimshuffle(1,0,2), 'valid', T(node.op.image_shape), T(node.op.filter_shape)).dimshuffle(1,0,2))]
        else:
            r = [core_conv1d_fft(top.dimshuffle(1,0,2), img.dimshuffle(1,0,2)[:,:,::-1], 'valid', T(node.op.filter_shape), T(node.op.image_shape))]
        return r

def conv1d_fft(input, filters, border_mode='valid', image_shape=None, filter_shape=None):
    return Symbol1DConv(border_mode,image_shape,filter_shape)(basic_ops.as_cuda_ndarray_variable(input), basic_ops.as_cuda_ndarray_variable(filters))

import scipy.optimize
_epsilon = 1e-2
from scipy.optimize import check_grad, approx_fprime
from numpy import sqrt
import numpy.random as npr

def check_grad(func, grad, x0, *args):
    """Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.

    Parameters
    ----------
    func : callable func(x0,*args)
        Function whose derivative is to be checked.
    grad : callable grad(x0, *args)
        Gradient of `func`.
    x0 : ndarray
        Points to check `grad` against forward difference approximation of grad
        using `func`.
    args : \*args, optional
        Extra arguments passed to `func` and `grad`.

    Returns
    -------
    err : float
        The square root of the sum of squares (i.e. the 2-norm) of the
        difference between ``grad(x0, *args)`` and the finite difference
        approximation of `grad` using func at the points `x0`.

    See Also
    --------
    approx_fprime

    Notes
    -----
    The step size used for the finite difference approximation is
    `sqrt(numpy.finfo(float).eps)`, which is approximately 1.49e-08.

    Examples
    --------
    >>> def func(x): return x[0]**2 - 0.5 * x[1]**3
    >>> def grad(x): return [2 * x[0], -1.5 * x[1]**2]
    >>> check_grad(func, grad, [1.5, -1.5])
    2.9802322387695312e-08

    """
    return sqrt(sum((grad(x0, *args) -
                     approx_fprime(x0, func, _epsilon, *args))**2))

if __name__=="__main__":
    obj = T.matrix()
    cufftop = theano.sandbox.cuda.fftconv.cufft(obj)
    f = theano.function([obj],cufftop)
    print "STAGE"
    inpd = np.array(
        [
            [1,2,3,4],
            [5,6,7,8]],'f')
    print f(inpd).__array__()

    SU = [2,10,50]
    SV = [5,10,15]

    pSU = np.prod(SU)
    pSV = np.prod(SV)

    u = T.tensor3()
    v = T.tensor3()
    u1 = basic_ops.as_cuda_ndarray_variable(u)
    v1 = basic_ops.as_cuda_ndarray_variable(v)
    w = conv1d_fft(u1,v1,'full',SU,SV)
    wsqr = (w*w).sum()
    r = theano.grad(wsqr,[u1,v1])
    f = theano.function([u,v],wsqr)
    g = theano.function([u,v],r)
    def func(x0):
        x0 = x0.astype('f')
        iu = x0[:pSU].reshape(SU)
        iv = x0[pSU:].reshape(SV)

        return f(iu,iv)

    def grad(x0):
        x0 = x0.astype('f')
        iu = x0[:pSU].reshape(SU)
        iv = x0[pSU:].reshape(SV)
        print iu,iv
        print f(iu,iv)
        g1 = g(iu,iv)
        print g1[0].__array__(), g1[1].__array__()
        return np.concatenate([g1[0].__array__().flatten(), g1[1].__array__().flatten()])

    npr.seed(12345)
    x0 = npr.ranf(pSU+pSV)-0.5
    print check_grad(func,grad,x0)


