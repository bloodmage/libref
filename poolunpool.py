import layerbase
import theano
import theano.tensor as T
from layerbase import Layer, dtypeX
class StruPool(Layer):
    def __init__(self, bs):
        base = bs.output.reshape((bs.output_shape[0], bs.output_shape[1], bs.output_shape[2]/2,2, bs.output_shape[3]/2,2))
        baser = base.dimshuffle(0,1,2,4,3,5).reshape((bs.output_shape[0]* bs.output_shape[1]* bs.output_shape[2]*bs.output_shape[3]/4,4))
        m, arg = T.max_and_argmax(baser,axis=1)
        self.output = m.reshape((bs.output_shape[0], bs.output_shape[1], bs.output_shape[2]/2, bs.output_shape[3]/2))
        self.output_shape = (bs.output_shape[0], bs.output_shape[1], bs.output_shape[2]/2, bs.output_shape[3]/2)
        self.unarg = arg.reshape((bs.output_shape[0], bs.output_shape[1], bs.output_shape[2]/2, bs.output_shape[3]/2))
        self.origshape = bs.output_shape

class StruUnpool(Layer):
    def __init__(self, base, lastpool, copies = None):
        if copies==None:
            copies = base.output_shape[1] / lastpool.origshape[1]
        arg = lastpool.unarg
        arg = arg.repeat(copies,axis=1)
        arg = arg.flatten()
        arg = arg + T.cast(T.arange(arg.shape[0])*4,'int32')
        resp = T.alloc(dtypeX(0), arg.shape[0]*4)
        resp = T.set_subtensor(resp[arg], base.output.flatten()) #Unpooling
        self.output = resp.reshape((lastpool.origshape[0], base.output_shape[1], lastpool.origshape[2]/2, lastpool.origshape[3]/2, 2,2)).dimshuffle(0,1,2,4,3,5).reshape((lastpool.origshape[0], base.output_shape[1], lastpool.origshape[2], lastpool.origshape[3]))
        self.output_shape = (lastpool.origshape[0], base.output_shape[1], lastpool.origshape[2], lastpool.origshape[3])
        print self.output_shape, lastpool.origshape, base.output_shape

from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMaxGrad
class StruPool(Layer):
    def __init__(self, bs):
        mp = max_pool_2d(bs.output, (2,2))
        self.output = mp
        self.output_shape = (bs.output_shape[0], bs.output_shape[1], bs.output_shape[2]/2, bs.output_shape[3]/2)

        gradop = DownsampleFactorMaxGrad((2,2), st=(2,2), padding=(0,0), ignore_border=False)
        self.upop = lambda v:gradop(bs.output, mp, v)
        self.origshape = bs.output_shape

class StruUnpool(Layer):
    def __init__(self, base, lastpool, copies = None):
        if copies==None:
            copies = base.output_shape[1] / lastpool.origshape[1]
        uout = []
        for i in range(copies):
            uout.append(lastpool.upop(base.output[:,i*lastpool.origshape[1]:(i+1)*lastpool.origshape[1]]))

        self.output = T.concatenate(uout, axis=1)
        self.output_shape = (lastpool.origshape[0], base.output_shape[1], lastpool.origshape[2], lastpool.origshape[3])
        print self.output_shape, lastpool.origshape, base.output_shape

import numpy as np
class StruSMPool(Layer):
    def __init__(self, bs, MAG = np.float(1)):
        #Power term
        powerterm = T.exp((bs.output-bs.output.max()) * MAG)
        valterm = powerterm * bs.output
        #Sum both
        shshape = (bs.output_shape[0], bs.output_shape[1], bs.output_shape[2]/2, 2, bs.output_shape[3]/2, 2)
        powerres = powerterm.reshape(shshape).sum(axis=(3,5))
        valres = valterm.reshape(shshape).sum(axis=(3,5))
        #Pooling result
        self.output = valres / powerres
        self.output_shape = (bs.output_shape[0], bs.output_shape[1], bs.output_shape[2]/2, bs.output_shape[3]/2)
        #Store for unpooling
        self.powerres = powerres
        self.valres = valres
        self.powerterm = powerterm
        self.valterm = valterm
        self.origshape = bs.output_shape

class StruSMUnpool(Layer):
    def __init__(self, base, lastpool):
        #if copies==None:
        copies = base.output_shape[1] / lastpool.origshape[1]
        uout = []
        for i in range(copies):
            uout.append((base.output[:,i*lastpool.origshape[1]:(i+1)*lastpool.origshape[1]] / lastpool.powerres).repeat(2,axis=2).repeat(2,axis=3) * lastpool.powerterm)

        self.output = T.concatenate(uout, axis=1)
        self.output_shape = (lastpool.origshape[0], base.output_shape[1], lastpool.origshape[2], lastpool.origshape[3])


