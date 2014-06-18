import numpy as np
import numpy.random as npr
try:
    import theano
    import theano.tensor as T
    import theano.tensor.nnet as nnet
    from theano.tensor.signal import downsample
    from theano.tensor.nnet import conv
    import theano.tensor.signal.conv as sconv
except:
    print "THEANO ERROR, THEANO-Based functions are disabled"
import os
import math

class safefile:
    def __init__(self,name):
        self.name = name
        self.mode = 0
        self.f = None

    def __enter__(self):
        if os.path.exists(self.name+'.tmp') and not os.path.exists(self.name):
            os.rename(self.name+'.tmp', self.name)
        else:
            try: os.unlink(self.name+'.tmp')
            except: pass
        return self

    def __nonzero__(self):
        return os.path.exists(self.name)
    
    def rb(self):
        self.f=file(self.name,'rb')
        return self.f

    def wb(self):
        self.f=file(self.name+'.tmp', 'wb')
        self.mode = 1
        return self.f

    def __exit__(self, type, value, tb):
        if type!=None:
            try: self.f.close()
            except: pass
            try: os.unlink(self.name+'.tmp')
            except: pass
        elif self.mode == 1:
            self.f.close()
            if not os.path.exists(self.name+'.tmp'):
                raise Exception("File not generated")
            try: os.unlink(self.name)
            except: pass
            os.rename(self.name+'.tmp', self.name)

class Layer: pass
class Param: pass
class VisLayer: pass

def nonlinear(input, nonlinear = 'tanh'):
    if nonlinear == 'tanh' or nonlinear == True:
        return T.tanh(input)
    elif nonlinear == 'rectifier':
        return input * (input > 0)
    elif nonlinear == 'sigmoid':
        return nnet.sigmoid(input)
    elif nonlinear == 'linear' or not nonlinear:
        return input
    else:
        raise Exception("Unknown nonlinear %s"%nonlinear)

class ConvLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, filter_shape, image_shape = None, isShrink = True, Nonlinear = "tanh", zeroone = False, inc=[0]):

        if isinstance(input, Layer):
            self.input = input.output 
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]

        fan_in = np.prod(filter_shape[1:])
        W_values = np.asarray(rng.uniform(
              low=-np.sqrt(0.5/fan_in),
              high=np.sqrt(0.5/fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv.conv2d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="valid" if isShrink else "full")
        
        self.output = nonlinear(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), Nonlinear)
        if zeroone:
            self.output = (self.output+1) * 0.5
        
        self.output_shape = (image_shape[0], filter_shape[0],
                image_shape[2]-filter_shape[2]+1 if isShrink else image_shape[2]+filter_shape[2]-1,
                image_shape[3]-filter_shape[3]+1 if isShrink else image_shape[3]+filter_shape[3]-1)
        
        self.params = [self.W, self.b]

        inc[0] = inc[0]+1
    
class ConvMaxoutLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, filter_shape, image_shape = None, isShrink = True, maxout_size = 5, inc=[0]):

        if isinstance(input, Layer):
            self.input = input.output
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]
        filter_shape = list(filter_shape)
        filter_shape[0] *= maxout_size
        #assert filter_shape[0] % maxout_size == 0

        fan_in = np.prod(filter_shape[1:])
        W_values = np.asarray(rng.uniform(
              low=-np.sqrt(0.5/fan_in),
              high=np.sqrt(0.5/fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv.conv2d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="valid" if isShrink else "full")

        self.output_shape = (image_shape[0], filter_shape[0]/maxout_size,
                image_shape[2]-filter_shape[2]+1 if isShrink else image_shape[2]+filter_shape[2]-1,
                image_shape[3]-filter_shape[3]+1 if isShrink else image_shape[3]+filter_shape[3]-1)
        
        self.output = T.max(T.reshape(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), (self.output_shape[0], self.output_shape[1], maxout_size, self.output_shape[2], self.output_shape[3])), axis=2)

        self.params = [self.W, self.b]

        inc[0] = inc[0]+1

class MLPConvLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, hidden_size, image_shape = None, inc = [0]):

        if isinstance(input, Layer):
            self.input = input.output
            if image_shape == None:
                image_shape = input.output_shape
        else:
            self.input = input

        #Dimshuffle to make a dotable plane
        filter_shape = (hidden_size, image_shape[1])
        fan_in = image_shape[1]
        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(0.5/fan_in),
            high=np.sqrt(0.5/fan_in),
            size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='Wmlpconv_%s'%inc[0])

        b_values = np.zeros((hidden_size,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='bmlpconv_%s'%inc[0])

        plane = self.input.dimshuffle(1, 0, 2, 3).reshape((image_shape[1], image_shape[0]*image_shape[2]*image_shape[3]))
        planeout = T.dot(self.W, plane) + self.b.dimshuffle(0, 'x')
        planeout = T.tanh(planeout) #Rectifier

        #Make a graphic size output
        self.output = planeout.reshape((hidden_size, image_shape[0], image_shape[2], image_shape[3])).dimshuffle(1, 0, 2, 3)
        self.output_shape = (image_shape[0], hidden_size, image_shape[2], image_shape[3])

        self.params = [self.W, self.b]

        inc[0] = inc[0]+1

class ConvKeepLayer(Layer, Param, VisLayer):
    def __init__(self, rng, input, filter_shape, image_shape = None, Nonlinear = "tanh", zeroone = False, inc=[0], dropout = False, dropoutrnd = None):

        if isinstance(input, Layer):
            self.input = input.output 
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]
        assert filter_shape[2]%2 == 1
        assert filter_shape[3]%2 == 1
        med = (filter_shape[2]-1)/2,(filter_shape[3]-1)/2

        fan_in = np.prod(filter_shape[1:])
        W_values = np.asarray(rng.uniform(
              low=-np.sqrt(0.5/fan_in),
              high=np.sqrt(0.5/fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv.conv2d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="full")
        #Get middle area
        conv_out = conv_out[:,:,med[0]:-med[0],med[1]:-med[1]]

        self.output = nonlinear(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), Nonlinear)
        if zeroone:
            self.output = (self.output+1) * 0.5
        self.output_shape = (image_shape[0], filter_shape[0], image_shape[2], image_shape[3])
        

        if not (dropout is False): #Embed a layerwise dropout layer
            self.output = LayerbasedDropOut(self, dropoutrnd, dropout).output
        
        self.params = [self.W, self.b]

        inc[0] = inc[0]+1

class Maxpool2DLayer(Layer):

    def __init__(self, input, max_pool_size = (2,2), ignore_border = False, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input

        self.output = downsample.max_pool_2d(self.input, max_pool_size, ignore_border)
        if not ignore_border:
            self.output_shape = (image_shape[0], image_shape[1], (image_shape[2]+max_pool_size[0]-1)/max_pool_size[0], (image_shape[3]+max_pool_size[1]-1)/max_pool_size[1])
        else:
            self.output_shape = (image_shape[0], image_shape[1], image_shape[2]/max_pool_size[0], image_shape[3]/max_pool_size[1])

class Maxpool2D1DLayer(Layer):

    def __init__(self, input, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        
        inputflat = T.reshape(self.input,(image_shape[0],image_shape[1],image_shape[2]*image_shape[3]))
        self.output = T.max(inputflat, axis=2)
        self.output_shape = (image_shape[0],image_shape[1])

class FullConnectLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, hidden_size, Nonlinear = True, reshape = None, input_shape = None, inc = [0]):

        if isinstance(input, Layer):
            self.input = input.output
            if input_shape==None:
                input_shape = input.output_shape
        else:
            self.input = input

        fan_in = input_shape[1]
        filter_shape = (input_shape[1], hidden_size)

        W_values = np.asarray(rng.uniform(
              low=-np.sqrt(0.5/fan_in),
              high=np.sqrt(0.5/fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='Wflat_%s'%inc[0])

        b_values = np.zeros((hidden_size,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='bflat_%s'%inc[0])
        
        self.output = T.dot(self.input, self.W) + self.b.dimshuffle('x', 0)
        if Nonlinear:
            self.output = T.tanh(self.output)
        if reshape:
            self.output = T.reshape(self.output, reshape)
            self.output_shape = reshape
        else:
            self.output_shape = (input_shape[0], hidden_size)
        self.reshape = reshape

        self.params = [self.W, self.b]
        inc[0] += 1

class DataLayer(Layer, VisLayer):

    def __init__(self, data, resp, batch_size):
        self.dataset = theano.shared(value=data, name='data')
        self.respset = theano.shared(value=resp, name='resp')
        self.output = T.tensor4('data')
        self.output_shape = (batch_size, data.shape[1], data.shape[2], data.shape[3])
        self.resp = T.tensor4('resp')
        self.resp_shape = (batch_size, resp.shape[1], resp.shape[2], resp.shape[3])
        self.n_batches = data.shape[0] / batch_size
        self.batch_size = batch_size

    def givens(self, index):
        return {self.output: self.dataset[index*self.batch_size: (index+1)*self.batch_size],
                self.resp: self.respset[index*self.batch_size: (index+1)*self.batch_size]}

    def givens_1(self, index_1):
        return {self.output: self.dataset[index_1: (index_1+self.batch_size)],
                self.resp: self.respset[index_1: (index_1+self.batch_size)]}

class SymbolDataLayer(Layer, VisLayer):

    def __init__(self, datashape, respshape, batch_size):
        self.output = T.tensor4('data')
        self.output_shape = (batch_size, datashape[1], datashape[2], datashape[3])
        if len(respshape)==4:
            self.resp = T.tensor4('resp')
            self.resp_shape = (batch_size, respshape[1], respshape[2], respshape[3])
        else:
            self.resp = T.matrix('resp')
            self.resp_shape = (batch_size, respshape[1])
        self.n_batches = datashape[0] / batch_size
        self.batch_size = batch_size
        self.data = self.output
        self.label = self.resp
    

class MaskedHengeLoss(Layer, VisLayer):

    def __init__(self,input,response):
        targets = response.resp
        mask = T.sgn(targets)
        antargets=T.switch(T.gt(targets,0),targets,1+targets)
        self.loss = self.hengeloss = T.sum((mask*(antargets-input.output)).clip(0,1e10))
        self.output = response.resp
        self.output_shape = response.resp_shape


class SquareLoss(Layer, VisLayer):

    def __init__(self,input,response,mask=None):
        targets = response.resp
        self.loss = self.squareloss = T.sum((targets-input.output)*(targets-input.output)*(1 if mask==None else mask))
        self.output = targets
        self.output_shape = response.resp_shape

class SSIMLoss(Layer, VisLayer):
    
    def __init__(self, input, response, image_shape = None, gkern = 5, gsigma = 1.5, c1 = 6.5025, c2 = 58.5225):
        if isinstance(input, Layer):
            self.input = input.output 
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape == response.resp_shape
        kernel = np.array([[np.exp((-x*x-y*y)*0.5/gsigma/gsigma)/(gsigma * np.sqrt(2*np.pi)) for x in range(-gkern, gkern+1)] for y in range(-gkern, gkern+1)],'f')
        KERNEL =  theano.shared(kernel, name='SSIM_KERNEL_%s_%s'%(gkern,gsigma))

        iflat = self.input.reshape((image_shape[0]*image_shape[1], image_shape[2], image_shape[3]))
        oflat = response.resp.reshape((image_shape[0]*image_shape[1], image_shape[2], image_shape[3]))
        iflatsqr = iflat * iflat
        oflatsqr = oflat * oflat
        crossflat = iflat * oflat

        iwindow = sconv.conv2d(iflat, KERNEL, image_shape = image_shape, filter_shape = (gkern*2+1, gkern*2+1), border_mode = 'full')[:,gkern:-gkern, gkern:-gkern].reshape(image_shape)
        isqrwin = sconv.conv2d(iflatsqr, KERNEL, image_shape = image_shape, filter_shape = (gkern*2+1, gkern*2+1), border_mode = 'full')[:,gkern:-gkern, gkern:-gkern].reshape(image_shape)
        owindow = sconv.conv2d(oflat, KERNEL, image_shape = image_shape, filter_shape = (gkern*2+1, gkern*2+1), border_mode = 'full')[:,gkern:-gkern, gkern:-gkern].reshape(image_shape)
        osqrwin = sconv.conv2d(oflatsqr, KERNEL, image_shape = image_shape, filter_shape = (gkern*2+1, gkern*2+1), border_mode = 'full')[:,gkern:-gkern, gkern:-gkern].reshape(image_shape)
        crosswin = sconv.conv2d(crossflat, KERNEL, image_shape = image_shape, filter_shape = (gkern*2+1, gkern*2+1), border_mode = 'full')[:,gkern:-gkern, gkern:-gkern].reshape(image_shape)

        vari = isqrwin - iwindow*iwindow
        varo = osqrwin - owindow*owindow
        cross = crosswin - iwindow*owindow

        SSIMblk = (2*iwindow*owindow + c1)*(2*cross + c2)/(iwindow*iwindow + owindow*owindow + c1)/(vari + varo + c2)
        SSIM = SSIMblk.mean()
        self.loss = 1-SSIM

        self.output = response.resp
        self.output_shape = response.resp_shape

class DropOut(Layer):

    def __init__(self,input,rnd,symboldropout=1):
        self.data=input.output
        self.output_shape=input.output_shape
        self.rnd=rnd.binomial(size=input.output_shape, dtype='float32')
        self.output=self.data*(1+symboldropout*(self.rnd*2-1))

class LayerbasedDropOut(Layer):

    def __init__(self,input,rnd,symboldropout=1):
        self.data=input.output
        self.output_shape=input.output_shape
        self.rnd=rnd.binomial(size=input.output_shape[0:2], dtype='float32')
        self.rnd = T.shape_padright(self.rnd, len(input.output_shape)-2)
        self.output = self.data*(1+symboldropout*(self.rnd*2-1))

class LogSoftmaxLayer(Layer, VisLayer):

    def __init__(self,input):
        self.output_shape = input.output_shape
        tdat = input.output.reshape((np.prod(self.output_shape[0:-1]),self.output_shape[-1]))
        tdat = tdat - tdat.max(axis=1,keepdims=True)
        tdat = tdat - T.log(T.exp(tdat).sum(axis=1,keepdims=True))

        self.output = tdat.reshape(self.output_shape)

class LabelLoss(Layer, VisLayer):

    def __init__(self,input,truth):
        self.loss = -(input.output * truth.resp).sum()
        self.output = truth.resp
        self.output_shape = truth.output_shape

def makesoftmaxlabel(raw, threshold = 50):
    processed = np.zeros((raw.shape[0]*raw.shape[2]*raw.shape[3],raw.shape[1]+1),'f')
    rawflat = raw.transpose((0,2,3,1)).reshape((-1,raw.shape[1]))
    labels = np.argmax(rawflat,axis=1)
    labels = np.where(rawflat[(np.arange(processed.shape[0]),labels)]>threshold,labels,raw.shape[1])
    processed[(np.arange(processed.shape[0]),labels)]=1
    processed = processed.reshape((raw.shape[0],raw.shape[2],raw.shape[3],raw.shape[1]+1)).transpose(0,3,1,2)

    return processed

class Model:

    def __init__(self, *layers):
        self.layers = layers

    def save(self, fileobj):
        idx = 0
        params = {}
        for i in self.layers:
            idx += 1
            if isinstance(i,Param):
                for j in i.params:
                    params['%s_%s'%(idx,j.name)] = j.get_value()
        np.savez(fileobj, **params)

    def load(self, fileobj):
        obj = np.load(fileobj)
        idx = 0
        for i in self.layers:
            idx += 1
            if isinstance(i,Param):
                for j in i.params:
                    try:
                        assert j.get_value().shape == obj['%s_%s'%(idx,j.name)].shape
                        j.set_value(obj['%s_%s'%(idx,j.name)])
                    except:
                        print "CANNOT SET",'%s_%s'%(idx,j.name)
    
    def outputs(self):
        p = []
        for i in self.layers:
            if isinstance(i,VisLayer):
                p.append(i.output)
        return p

    def params(self):
        p = []
        for i in self.layers:
            if isinstance(i,Param):
                p += i.params
        return p
    
    def paramlayers(self):
        for i in self.layers:
            if isinstance(i,Param):
                yield i

    def pmomentum(self):
        p = self.params()
        q = []
        for i in p:
            init = np.zeros_like(i.get_value())
            q.append(theano.shared(init, name=i.name+'_momentum'))
        return q

def DrawPatch(block, blknorm = True):
    EPS = 1e-10
    flatblk = np.copy(block.reshape((-1,block.shape[2],block.shape[3])))
    if blknorm:
        flatblk = (flatblk - np.min(flatblk)) / (np.max(flatblk) - np.min(flatblk)+EPS)
    else:
        flatblk = (flatblk-np.min(flatblk, axis=(1,2), keepdims=True)) / (np.max(flatblk, axis=(1,2), keepdims=True) - np.min(flatblk, axis=(1,2), keepdims=True)+EPS)

    width = math.ceil(math.sqrt(flatblk.shape[0]))
    height = (flatblk.shape[0] + width - 1) // width
    canvas = np.zeros((height*block.shape[2]+height-1,width*block.shape[3]+width-1),'f')
    for i in range(flatblk.shape[0]):
        y = i // width
        x = i % width
        canvas[y*block.shape[2]+y:(y+1)*block.shape[2]+y,x*block.shape[3]+x:(x+1)*block.shape[3]+x] = flatblk[i]
    return np.array(canvas*255,np.uint8)

def MaxBlend(image, red = None, green = None, blue = None):
    if len(image.shape)==2:
        #Extend image array to 3D
        image=np.repeat(image,3,1).reshape(image.shape+(3,))
    else:
        image=np.copy(image)
    for comp, idx in zip([red,green,blue],[0,1,2]):
        if comp==None: continue
        image[:,:,idx] = np.maximum(image[:,:,idx], comp)

    return image

def DrawMaskedPatch(block, mask, blknorm = True):
    EPS = 1e-10
    flatblk = np.copy(block.reshape((-1,block.shape[2],block.shape[3])))
    mask = np.clip(mask,0,1)
    if blknorm:
        flatblk = (flatblk - np.min(flatblk)) / (np.max(flatblk) - np.min(flatblk)+EPS)
    else:
        flatblk = (flatblk-np.min(flatblk, axis=(1,2), keepdims=True)) / (np.max(flatblk, axis=(1,2), keepdims=True) - np.min(flatblk, axis=(1,2), keepdims=True)+EPS)

    width = math.ceil(math.sqrt(flatblk.shape[0]))
    height = (flatblk.shape[0] + width - 1) // width
    canvas = np.zeros((height*block.shape[2]+height-1,width*block.shape[3]+width-1,3),'f')
    for i in range(flatblk.shape[0]):
        y = i // width
        x = i % width
        canvas[y*block.shape[2]+y:(y+1)*block.shape[2]+y,x*block.shape[3]+x:(x+1)*block.shape[3]+x] = MaxBlend(flatblk[i],None,mask,None)
    return np.array(canvas*255,np.uint8)

def shuffle_in_unison_inplace(rng, a, b):
    assert len(a) == len(b)
    p = rng.permutation(len(a))
    return a[p], b[p]

