import numpy as np
import theano.tensor as T
import theano.sandbox
import theano.gof
import theano

from theano import config
from theano.sandbox.neighbours import images2neibs
from layerbase import Layer, Param, VisLayer, LayerbasedDropOut, VisSamerank
from theano.tensor.nnet import conv
INF = 1e10
b3tensor = T.TensorType(dtype = theano.config.floatX, broadcastable = [])

def max_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Theano's max pooling op only supports pool_stride = pool_shape
    so here we have a graph that does max pooling with strides

    Parameters
    ----------
    bc01 : theano tensor
        minibatch in format (batch size, channels, rows, cols)
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano

    Returns
    -------
    pooled : theano tensor
        The output of pooling applied to `bc01`

    See Also
    --------
    max_pool_c01b : Same functionality but with ('c', 0, 1, 'b') axes
    sandbox.cuda_convnet.pool.max_pool_c01b : Same functionality as
        `max_pool_c01b` but GPU-only and considerably faster.
    mean_pool : Mean pooling instead of max pooling
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    assert pr <= r
    assert pc <= c

    name = bc01.name
    if name is None:
        name = 'anon_bc01'

    if pool_shape == pool_stride:
        mx = max_pool_2d(bc01, pool_shape, False)
        mx.name = 'max_pool('+name+')'
        return mx

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc


    wide_infinity = T.alloc(T.constant(-np.inf, dtype=config.floatX),
                            bc01.shape[0],
                            bc01.shape[1],
                            required_r,
                            required_c)

    bc01 = T.set_subtensor(wide_infinity[:, :, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,
                       :,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs]
            cur.name = ('max_pool_cur_' + bc01.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = ('max_pool_mx_' + bc01.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))

    mx.name = 'max_pool('+name+')'

 
    return mx


def dtypeX(val):
    return np.dtype(theano.config.floatX).type(val)

class StacksampleFractal(Layer):
    
    def __init__(self, input, input_shape = None, feedval = 0.0):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape

        #Only square image allowed
        assert input_shape[2]==input_shape[3]

        #Extend one pixel at each direction
        shapeext = input_shape[0], input_shape[1], input_shape[2]+2, input_shape[3]+2
        inputext = T.alloc(dtypeX(-INF), *shapeext)

        inputext = T.set_subtensor(inputext[:,:,1:input_shape[2]+1,1:input_shape[3]+1], self.input)
        
        output_cmb = max_pool(inputext, (3,3), (1,1), shapeext[2:])
        self.output_cmb = output_cmb
        #Separate output to 4 channels
        c00 = output_cmb[:,:,::2,::2]
        c01 = output_cmb[:,:,::2,1::2]
        c10 = output_cmb[:,:,1::2,::2]
        c11 = output_cmb[:,:,1::2,1::2]
        self.one_channel = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2

        #Combine, 2 conditions: even/odd
        if (input_shape[2]&1)==0:
            joined = T.concatenate([c00,c01,c10,c11], axis=0)
        else:
            joined = T.alloc(dtypeX(feedval), *((input_shape[0]*4,)+self.one_channel[1:4]))
            joined = T.set_subtensor(joined[0:self.one_channel[0],:,:,:], c00)
            joined = T.set_subtensor(joined[self.one_channel[0]:self.one_channel[0]*2,:,:,:-1], c01)
            joined = T.set_subtensor(joined[self.one_channel[0]*2:self.one_channel[0]*3,:,:-1,:], c10)
            joined = T.set_subtensor(joined[self.one_channel[0]*3:self.one_channel[0]*4,:,:-1,:-1], c11)

        self.output = joined
        self.output_shape = input_shape[0]*4, self.one_channel[1], self.one_channel[2], self.one_channel[3]

class SymbolStacksampleFractal(StacksampleFractal):
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.one_channel = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2

class DestacksampleFractal(Layer):

    def __init__(self, input, stacksamplelayer, input_shape = None):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape
        assert isinstance(stacksamplelayer, StacksampleFractal)
        
        #Insert back by shape
        if (stacksamplelayer.input_shape[2]&1)==0:
            self.output_shape = stacksamplelayer.input_shape[0], input_shape[1], input_shape[2]*2, input_shape[3]*2
        else:
            self.output_shape = stacksamplelayer.input_shape[0], input_shape[1], input_shape[2]*2-1, input_shape[3]*2-1
        self.output = T.alloc(dtypeX(0.0), *self.output_shape)
        #print self.output_shape

        c00 = self.input[0:stacksamplelayer.one_channel[0]]
        c01 = self.input[stacksamplelayer.one_channel[0]:stacksamplelayer.one_channel[0]*2]
        c10 = self.input[stacksamplelayer.one_channel[0]*2:stacksamplelayer.one_channel[0]*3]
        c11 = self.input[stacksamplelayer.one_channel[0]*3:stacksamplelayer.one_channel[0]*4]

        if (stacksamplelayer.input_shape[2]&1)==0:
            self.output = T.set_subtensor(self.output[:,:,::2,::2], c00)
            self.output = T.set_subtensor(self.output[:,:,::2,1::2], c01)
            self.output = T.set_subtensor(self.output[:,:,1::2,::2], c10)
            self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11)
        else:
            self.output = T.set_subtensor(self.output[:,:,::2,::2], c00)
            self.output = T.set_subtensor(self.output[:,:,::2,1::2], c01[:,:,:,:-1])
            self.output = T.set_subtensor(self.output[:,:,1::2,::2], c10[:,:,:-1,:])
            self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11[:,:,:-1,:-1])

class StacklayerFractal(Layer):
    
    def __init__(self, input, input_shape = None, feedval = 0.0):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape

        #Only square image allowed
        assert input_shape[2]==input_shape[3]

        #Extend one pixel at each direction
        shapeext = input_shape[0], input_shape[1], input_shape[2]+2, input_shape[3]+2
        inputext = T.alloc(dtypeX(-INF), *shapeext)

        inputext = T.set_subtensor(inputext[:,:,1:input_shape[2]+1,1:input_shape[3]+1], self.input)
        
        output_cmb = max_pool(inputext, (3,3), (1,1), shapeext[2:])
        self.output_cmb = output_cmb
        #Separate output to 4 channels
        c00 = output_cmb[:,:,::2,::2]
        c01 = output_cmb[:,:,::2,1::2]
        c10 = output_cmb[:,:,1::2,::2]
        c11 = output_cmb[:,:,1::2,1::2]
        self.one_channel = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2

        #Combine, 2 conditions: even/odd
        if (input_shape[2]&1)==0:
            joined = T.concatenate([c00,c01,c10,c11], axis=1)
        else:
            joined = T.alloc(dtypeX(feedval), *((input_shape[0]*4,)+self.one_channel[1:4]))
            joined = T.set_subtensor(joined[:,0:self.one_channel[1],:,:], c00)
            joined = T.set_subtensor(joined[:,self.one_channel[1]:self.one_channel[1]*2,:,:-1], c01)
            joined = T.set_subtensor(joined[:,self.one_channel[1]*2:self.one_channel[1]*3,:-1,:], c10)
            joined = T.set_subtensor(joined[:,self.one_channel[1]*3:self.one_channel[1]*4,:-1,:-1], c11)

        self.output = joined
        self.output_shape = input_shape[0], self.one_channel[1]*4, self.one_channel[2], self.one_channel[3]

class SymbolStacklayerFractal(StacklayerFractal):
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.one_channel = input_shape[0], input_shape[1]*4, (input_shape[2]+1)/2, (input_shape[3]+1)/2

class DestacklayerFractal(Layer):

    def __init__(self, input, stacksamplelayer, input_shape = None):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape
        assert isinstance(stacksamplelayer, StacklayerFractal)
        
        #Insert back by shape
        if (stacksamplelayer.input_shape[2]&1)==0:
            self.output_shape = input_shape[0], stacksamplelayer.input_shape[1], input_shape[2]*2, input_shape[3]*2
        else:
            self.output_shape = input_shape[0], stacksamplelayer.input_shape[1], input_shape[2]*2-1, input_shape[3]*2-1
        self.output = T.alloc(dtypeX(0.0), *self.output_shape)
        #print self.output_shape

        c00 = self.input[:,0:stacksamplelayer.input_shape[1]]
        c01 = self.input[:,stacksamplelayer.input_shape[1]:stacksamplelayer.input_shape[1]*2]
        c10 = self.input[:,stacksamplelayer.input_shape[1]*2:stacksamplelayer.input_shape[1]*3]
        c11 = self.input[:,stacksamplelayer.input_shape[1]*3:stacksamplelayer.input_shape[1]*4]

        if (stacksamplelayer.input_shape[2]&1)==0:
            self.output = T.set_subtensor(self.output[:,:,::2,::2], c00)
            self.output = T.set_subtensor(self.output[:,:,::2,1::2], c01)
            self.output = T.set_subtensor(self.output[:,:,1::2,::2], c10)
            self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11)
        else:
            self.output = T.set_subtensor(self.output[:,:,::2,::2], c00)
            self.output = T.set_subtensor(self.output[:,:,::2,1::2], c01[:,:,:,:-1])
            self.output = T.set_subtensor(self.output[:,:,1::2,::2], c10[:,:,:-1,:])
            self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11[:,:,:-1,:-1])

class ShrinkshapeMeanFractal(Layer, VisSamerank):

    def __init__(self,input,input_shape = None):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape

        #Only square image allowed
        assert input_shape[2]==input_shape[3]

        #Extend one pixel at each direction
        shapeext = input_shape[0], input_shape[1], input_shape[2]+2, input_shape[3]+2
        inputext = T.alloc(dtypeX(-INF), *shapeext)

        inputext = T.set_subtensor(inputext[:,:,1:input_shape[2]+1,1:input_shape[3]+1], self.input)
        self.output = images2neibs(inputext, (3,3), (2,2), 'ignore_borders').mean(axis=-1)
        
        self.output_shape = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2

class ShrinkshapeFractal(Layer):

    def __init__(self,input,input_shape = None):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape

        #Only square image allowed
        assert input_shape[2]==input_shape[3]

        #Extend one pixel at each direction
        shapeext = input_shape[0], input_shape[1], input_shape[2]+1, input_shape[3]+1
        inputext = T.alloc(dtypeX(-INF), *shapeext)

        inputext = T.set_subtensor(inputext[:,:,1:input_shape[2]+1,1:input_shape[3]+1], self.input)
        
        self.output = max_pool(inputext, (3,3), (2,2), shapeext[2:])
        self.output_shape = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2

class ExpandshapeFractal(Layer):

    def __init__(self, input, shrinksamplelayer, input_shape=None, calibrate = True, smallestexpand = False):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape
         
        assert isinstance(shrinksamplelayer, ShrinkshapeFractal)

        if calibrate:
            cali=2
        else:
            cali=0

        self.output_shape = input_shape[0], input_shape[1]+cali, shrinksamplelayer.input_shape[2], shrinksamplelayer.input_shape[3]

        output = T.alloc(dtypeX(0.0), *self.output_shape)

        #Expand data 4 fold
        if (shrinksamplelayer.input_shape[2]&1)==0:
            output = T.set_subtensor(output[:,:input_shape[1],::2,::2], self.input)
            if not smallestexpand:
                output = T.set_subtensor(output[:,:input_shape[1],::2,1::2], self.input)
                output = T.set_subtensor(output[:,:input_shape[1],1::2,::2], self.input)
                output = T.set_subtensor(output[:,:input_shape[1],1::2,1::2], self.input)
        else:
            output = T.set_subtensor(output[:,:input_shape[1],::2,::2], self.input)
            if not smallestexpand:
                output = T.set_subtensor(output[:,:input_shape[1],::2,1::2], self.input[:,:,:,:-1])
                output = T.set_subtensor(output[:,:input_shape[1],1::2,::2], self.input[:,:,:-1,:])
                output = T.set_subtensor(output[:,:input_shape[1],1::2,1::2], self.input[:,:,:-1,:-1])
        
        #Feed calibrate data
        if calibrate:
            #HACK, strange
            dval = T.alloc(dtypeX(1.0), input_shape[0], shrinksamplelayer.input_shape[2], shrinksamplelayer.input_shape[3]/2)
            output = T.set_subtensor(output[:,input_shape[1],:,1::2], dval)
            dval = T.alloc(dtypeX(1.0), input_shape[0], shrinksamplelayer.input_shape[2]/2, shrinksamplelayer.input_shape[3])
            output = T.set_subtensor(output[:,input_shape[1]+1,1::2], dval)

        self.output = output

class AggregationLayer(Layer):

    def __init__(self, *layers):

        channels = 0
        for i in layers:
            assert isinstance(i, Layer)
            channels += i.output_shape[1]

        self.output_shape = layers[0].output_shape[0], channels, layers[0].output_shape[2], layers[0].output_shape[3]
        self.output = T.alloc(dtypeX(0.0), *self.output_shape)
        channels = 0
        for i in layers:
            Layer.linkstruct[i].append(self)
            self.output = T.set_subtensor(self.output[:,channels:channels+i.output_shape[1]], i.output)
            channels += i.output_shape[1]

from layerbase import ConvKeepLayer

if __name__=="__main__":
    import numpy as np
    import theano

    a=np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]],'f')

    #Make subsample fractal test layer
    inp = T.tensor4("inp")
    l1 = StacksampleFractal(inp, (1,1,4,4))
    l2 = DestacksampleFractal(l1, l1)
    f = theano.function([inp], [l1.output, l2.output])
    print f(a)

    l1 = ShrinkshapeFractal(inp, (1,1,4,4))
    l2 = ExpandshapeFractal(l1, l1)
    f = theano.function([inp], [l1.output, l2.output])
    print f(a)

    
