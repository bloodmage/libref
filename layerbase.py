import numpy as np
import numpy.random as npr
try:
    import theano
    import theano.tensor as T
    import theano.tensor.nnet as nnet
    from theano.tensor.signal import downsample
    from theano.tensor.nnet import conv
    try: import theano.tensor.signal.conv as sconv
    except: print "SIGNAL CONV DISABLED"
    try: import theano.sandbox.cuda.dnn as dconv
    except: print "CUDNN CONV DISABLED"
    try: import theano.sandbox.cuda as cuda
    except: print "SANDBOX IMPORT FAILED"
except:
    print "THEANO ERROR, THEANO-Based functions are disabled"
import os
import math
import shutil

THEANO_CONV = 0
FFT_CONV = 1
CUDNN_CONV = 2
GPUCORR_CONV = 3

convmode = CUDNN_CONV

def dtypeX(val):
    return np.dtype(theano.config.floatX).type(val)

CachedResults = {}
def CachedInst(inststr,efunc = None):
    global CachedResults
    if inststr not in CachedResults:
        if efunc==None:
            CachedResults[inststr] = eval(inststr)
        else:
            CachedResults[inststr] = efunc()
    return CachedResults[inststr]

def setconvmode(mode):
    global convmode
    convmode = mode

def conv2d(input,filters,image_shape=None,filter_shape=None,border_mode='valid'):
    global convmode
    if convmode==CUDNN_CONV:
        if border_mode == 'same':
            return dconv.dnn_conv(input,filters,border_mode=(filter_shape[2]/2, filter_shape[3]/2), direction_hint='forward!')
    if border_mode=='same':
        allocspace = T.alloc(0.0, image_shape[0], image_shape[1], image_shape[2]+filter_shape[2]-1, image_shape[3]+filter_shape[3]-1)
        allocspace = T.patternbroadcast(allocspace, (False,)*4)
        allocspace = T.set_subtensor(allocspace[:,:,filter_shape[2]/2:filter_shape[2]/2+image_shape[2],filter_shape[3]/2:filter_shape[3]/2+image_shape[3]],input)
        border_mode='valid'
    else:
        allocspace = input
    if convmode==THEANO_CONV:
        return conv.conv2d(allocspace,filters,image_shape=image_shape,filter_shape=filter_shape,border_mode=border_mode)
    elif convmode==CUDNN_CONV:
        return dconv.dnn_conv(allocspace,filters,border_mode=border_mode, direction_hint='forward!')
    elif convmode==GPUCORR_CONV:
        import theano.sandbox.cuda.blas
        return theano.sandbox.cuda.blas.GpuCorrMM(border_mode)(allocspace,cuda.basic_ops.gpu_contiguous(filters[:,:,::-1,::-1]))
    elif convmode==FFT_CONV:
        if border_mode=='full':
            allocspace = T.alloc(0.0, image_shape[0], image_shape[1], image_shape[2]+filter_shape[2]*2-2, image_shape[3]+filter_shape[3]*2-2)
            allocspace = T.patternbroadcast(allocspace, (False,)*4)
            allocspace = T.set_subtensor(allocspace[:,:,filter_shape[2]-1:filter_shape[2]-1+image_shape[2],filter_shape[3]-1:filter_shape[3]-1+image_shape[3]],input)
            border_mode='valid'
        import fftconv
        return fftconv.conv2d_fft(allocspace,filters,image_shape=image_shape,filter_shape=filter_shape)
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
        #Directory based batch store
        if not os.path.exists(self.name):
            os.mkdir(self.name)
        if not os.path.isdir(self.name):
            os.rename(self.name, self.name+'.tmp')
            os.mkdir(self.name)
            shutil.copyfile(self.name+'.tmp', os.path.join(self.name, self.name))
        #Find latest idnumber
        idnames = [int(i[i.rfind('_')+1:]) for i in os.listdir(self.name) if i.rfind('_')!=-1]
        idnames.append(0)
        largestnum = max(idnames)+1
        self.largestnum = largestnum
        return self

    def __nonzero__(self):
        exists = os.path.exists(os.path.join(self.name,self.name))
        if not exists:
            print "NEW MODEL TO LOAD"
        return exists
    
    def rb(self):
        self.f=file(os.path.join(self.name,self.name),'rb')
        return self.f

    def wb(self):
        self.f=file(os.path.join(self.name,self.name+'.tmp'),'wb')
        self.mode = 1
        return self.f

    def __exit__(self, type, value, tb):
        try: self.f.close()
        except: pass
        if type!=None:
            try: os.unlink(os.path.join(self.name,self.name+'.tmp'))
            except: pass
        elif self.mode == 1:
            if not os.path.exists(os.path.join(self.name,self.name+'.tmp')):
                print ("Warning: File not generated")
                return
            if os.path.exists(os.path.join(self.name,self.name)):
                try: os.rename(os.path.join(self.name,self.name),os.path.join(self.name,self.name+'_'+str(self.largestnum)))
                except:
                    print "RENAME FAIL, IGNORE"
                    return
                if (self.largestnum-1)%100!=0:
                    os.unlink(os.path.join(self.name,self.name+'_'+str(self.largestnum-1)))
            os.rename(os.path.join(self.name,self.name+'.tmp'),os.path.join(self.name,self.name))

            #If need to remove last id

            self.largestnum+=1

import collections
class Layer:
    linkstruct = collections.defaultdict(lambda:[])

class Param: pass
class VisLayer:
    def setomit(self):
        self.omitvis = True
        return self
class LossLayer: pass
class VisSamerank: pass
class NParam: pass
class CMomentum: pass

def nonlinear(input, nonlinear = 'tanh'):
    if nonlinear == 'tanh' or nonlinear == True:
        return T.tanh(input)
    elif nonlinear == 'ftanh':
        return input / (1+abs(input))
    elif nonlinear == 'rectifier':
        return T.switch(input<0, 0, input)#input * (input > 0)
    elif nonlinear == 'sigmoid':
        return nnet.sigmoid(input)
    elif nonlinear == 'fsigmoid':
        return (2+input+abs(input))/(4+2*abs(input))
    elif nonlinear == 'softplus':
        return nnet.softplus(input)
    elif nonlinear == 'linear' or not nonlinear:
        return input
    else:
        raise Exception("Unknown nonlinear %s"%nonlinear)

class ConvLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, filter_shape, image_shape = None, isShrink = True, Nonlinear = "tanh", zeroone = False, inc=[0], shareLayer = None):

        if isinstance(input, Layer):
            self.input = input.output 
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]

        fan_in = np.prod(filter_shape[1:])
        
        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.uniform(
                  low=-np.sqrt(0.5/fan_in),
                  high=np.sqrt(0.5/fan_in),
                  size=filter_shape), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv2d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="valid" if isShrink else "full")
        
        self.output = nonlinear(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), Nonlinear)
        if zeroone:
            self.output = (self.output+1) * 0.5
        
        self.output_shape = (image_shape[0], filter_shape[0],
                image_shape[2]-filter_shape[2]+1 if isShrink else image_shape[2]+filter_shape[2]-1,
                image_shape[3]-filter_shape[3]+1 if isShrink else image_shape[3]+filter_shape[3]-1)
        
        if shareLayer==None:
            self.params = [self.W, self.b]
        else:
            self.params = []

        inc[0] = inc[0]+1
    
class ConvMaxoutLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, filter_shape, image_shape = None, isShrink = True, maxout_size = 5, inc=[0], shareLayer = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]
        filter_shape = list(filter_shape)
        filter_shape[0] *= maxout_size
        #assert filter_shape[0] % maxout_size == 0

        fan_in = np.prod(filter_shape[1:])
        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.uniform(
                  low=-np.sqrt(0.5/fan_in),
                  high=np.sqrt(0.5/fan_in),
                  size=filter_shape), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv2d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="valid" if isShrink else "full")

        self.output_shape = (image_shape[0], filter_shape[0]/maxout_size,
                image_shape[2]-filter_shape[2]+1 if isShrink else image_shape[2]+filter_shape[2]-1,
                image_shape[3]-filter_shape[3]+1 if isShrink else image_shape[3]+filter_shape[3]-1)
        
        self.output = T.max(T.reshape(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), (self.output_shape[0], self.output_shape[1], maxout_size, self.output_shape[2], self.output_shape[3])), axis=2)

        if shareLayer!=None:
            self.params = [self.W, self.b]
        else:
            self.params = []

        inc[0] = inc[0]+1

class MLPConvLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, hidden_size, image_shape = None, inc = [0], shareLayer = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape == None:
                image_shape = input.output_shape
        else:
            self.input = input

        #Dimshuffle to make a dotable plane
        filter_shape = (hidden_size, image_shape[1])
        fan_in = image_shape[1]
        
        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
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
        
        if shareLayer!=None:
            self.params = []
        else:
            self.params = [self.W, self.b]

        inc[0] = inc[0]+1

class ConvKeepLayer(Layer, Param, VisLayer):
    def __init__(self, rng, input, filter_shape, image_shape = None, Nonlinear = "tanh", zeroone = False, inc=[0], dropout = False, dropoutrnd = None, shareLayer = None, through = None, throughend = None):

        if isinstance(input, Layer):
            self.input = input.output 
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]
        assert filter_shape[2]%2 == 1
        assert filter_shape[3]%2 == 1
        med = (filter_shape[2]-1)/2,(filter_shape[3]-1)/2

        fan_in = np.prod(filter_shape[1:])

        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.uniform(
                  low=-np.sqrt(0.5/fan_in),
                  high=np.sqrt(0.5/fan_in),
                  size=filter_shape), dtype=theano.config.floatX)
            #if through!=None:
            #    for i in range(filter_shape[0] if throughend==None else throughend):
            #        W_values[i,through+i,(filter_shape[2]-1)/2,(filter_shape[3]-1)/2]=1
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv2d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="same")
        #Get middle area
        #conv_out = conv_out[:,:,med[0]:-med[0],med[1]:-med[1]]
        if through!=None:
            if throughend==None:
                conv_out = conv_out + self.input[:,through:through+filter_shape[0]]
            else:
                conv_out = T.inc_subtensor(conv_out[:,0:throughend], self.input[:,through:throughend+through])

        self.output = nonlinear(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), Nonlinear)
        if zeroone:
            self.output = (self.output+1) * 0.5
        self.output_shape = (image_shape[0], filter_shape[0], image_shape[2], image_shape[3])
        

        if not (dropout is False): #Embed a layerwise dropout layer
            if isinstance(dropout, tuple):
                self.output = dropout[0](self, dropoutrnd, dropout[1]).output
            else:
                self.output = LayerbasedDropOut(self, dropoutrnd, dropout).output
        
        if shareLayer!=None:
            self.params = []
        else:
            self.params = [self.W, self.b]
    
        inc[0] = inc[0]+1

class ConvMaxoutKeepLayer(Layer, Param, VisLayer):
    def __init__(self, rng, input, filter_shape, image_shape = None, MaxGroups = 5, zeroone = False, inc=[0], dropout = False, dropoutrnd = None, shareLayer = None, through = None, throughend = None):

        if isinstance(input, Layer):
            self.input = input.output 
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]
        assert filter_shape[2]%2 == 1
        assert filter_shape[3]%2 == 1
        med = (filter_shape[2]-1)/2,(filter_shape[3]-1)/2

        fan_in = np.prod(filter_shape[1:])
        grp_filter_shape = (filter_shape[0]*MaxGroups,)+filter_shape[1:]

        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.uniform(
                  low=-np.sqrt(0.5/fan_in),
                  high=np.sqrt(0.5/fan_in),
                  size=grp_filter_shape), dtype=theano.config.floatX)
            #if through!=None:
            #    for i in range(filter_shape[0] if throughend==None else throughend):
            #        W_values[i,through+i,(filter_shape[2]-1)/2,(filter_shape[3]-1)/2]=1
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

            b_values = np.zeros((grp_filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv2d(self.input, self.W,
                filter_shape=grp_filter_shape, image_shape=image_shape, border_mode="same")
        #Get middle area
        #conv_out = conv_out[:,:,med[0]:-med[0],med[1]:-med[1]]
        if through!=None:
            if throughend==None:
                conv_out = conv_out + self.input[:,through:through+filter_shape[0]]
            else:
                conv_out = T.inc_subtensor(conv_out[:,0:throughend], self.input[:,through:throughend+through])

        self.output = T.max((conv_out + self.b.dimshuffle('x', 0, 'x', 'x')).reshape((image_shape[0],filter_shape[0],MaxGroups, image_shape[2], image_shape[3])), 2)
        if zeroone:
            self.output = (self.output+1) * 0.5
        self.output_shape = (image_shape[0], filter_shape[0], image_shape[2], image_shape[3])
        

        if not (dropout is False): #Embed a layerwise dropout layer
            if isinstance(dropout, tuple):
                self.output = dropout[0](self, dropoutrnd, dropout[1]).output
            else:
                self.output = LayerbasedDropOut(self, dropoutrnd, dropout).output
        
        if shareLayer!=None:
            self.params = []
        else:
            self.params = [self.W, self.b]
    
        inc[0] = inc[0]+1


class ConcentrateConvKeepLayer(Layer, Param, VisLayer):
    def __init__(self, rng, input, concentrate, filter_shape, image_shape = None, Nonlinear = "tanh", zeroone = False, inc=[0], dropout = False, dropoutrnd = None, shareLayer = None, through = None, throughend = None):

        if isinstance(input, Layer):
            self.input = input.output 
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]
        assert filter_shape[2]%2 == 1
        assert filter_shape[3]%2 == 1
        med = (filter_shape[2]-1)/2,(filter_shape[3]-1)/2

        fan_in = np.prod(filter_shape[1:])

        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.uniform(
                  low=-np.sqrt(0.5/fan_in),
                  high=np.sqrt(0.5/fan_in),
                  size=filter_shape), dtype=theano.config.floatX)
            if through!=None:
                for i in range(filter_shape[0] if throughend==None else throughend):
                    W_values[i,through+i,(filter_shape[2]-1)/2,(filter_shape[3]-1)/2]=1
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])
        
        npconc = np.array(concentrate, dtype=theano.config.floatX)
        concentrate = theano.shared(value=npconc)
        concentrate = concentrate.reshape((1,1)+npconc.shape)

        conv_out = conv2d(self.input, self.W * concentrate,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="same")

        self.output = nonlinear(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), Nonlinear)
        if zeroone:
            self.output = (self.output+1) * 0.5
        self.output_shape = (image_shape[0], filter_shape[0], image_shape[2], image_shape[3])
        

        if not (dropout is False): #Embed a layerwise dropout layer
            if isinstance(dropout, tuple):
                self.output = dropout[0](self, dropoutrnd, dropout[1]).output
            else:
                self.output = LayerbasedDropOut(self, dropoutrnd, dropout).output
        
        if shareLayer!=None:
            self.params = []
        else:
            self.params = [self.W, self.b]
    
        inc[0] = inc[0]+1

class MulNonlinearLayer(Layer,VisLayer,VisSamerank):

    def __init__(self, input, leftnonlinear=False, rightnonlinear=False, outnonlinear='tanh'):

        assert isinstance(input, Layer)
        assert input.output_shape[1]%2==0
        Layer.linkstruct[input].append(self)

        self.input = input.output
        image_shape = input.output_shape

        self.linput = nonlinear(self.input[:,:image_shape[1]/2], leftnonlinear)
        self.rinput = nonlinear(self.input[:,image_shape[1]/2:], rightnonlinear)
        
        self.output = nonlinear(self.linput*self.rinput, outnonlinear)
        self.output_shape = image_shape[0], image_shape[1]/2, image_shape[2], image_shape[3]

class MulMuxLayer(Layer, VisSamerank):
    def __init__(self, input1, input2):
        assert input1.output_shape == input2.output_shape
        Layer.linkstruct[input1].append(self)
        Layer.linkstruct[input2].append(self)
        self.output = input1.output*input2.output
        self.output_shape = input1.output_shape

class AddMuxLayer(Layer, VisSamerank):
    def __init__(self, input1, input2):
        assert input1.output_shape == input2.output_shape
        Layer.linkstruct[input1].append(self)
        Layer.linkstruct[input2].append(self)
        self.output = input1.output+input2.output
        self.output_shape = input1.output_shape

def MulConvHelper(convlayer):
    if len(convlayer.params)==2:
        w,b = convlayer.params
        wval = w.get_value()
        bval = b.get_value()
        wval[convlayer.output_shape[1]/2:]=0
        bval[convlayer.output_shape[1]/2:]=1
        w.set_value(wval)
        b.set_value(bval)
    return convlayer

class LRN_AcrossMap(Layer,VisSamerank):
    
    def __init__(self, input, across, alpha = 1.0, beta = 1.0):
        assert across%2==1
        Layer.linkstruct[input].append(self)
        self.output_shape = input.output_shape
        kernel = np.zeros((self.output_shape[1],self.output_shape[1]),theano.config.floatX)
        for i in range(self.output_shape[1]):
            kernel[max(0,i-across/2):min(self.output_shape[1],i+across/2+1),i]=1.0
        self.k = theano.shared(value=kernel,name="LRN_kern_%s_%s"%(self.output_shape[1],across))
        
        osqr = input.output*input.output
        lpart = T.reshape(T.dot(T.reshape(osqr.dimshuffle(0,2,3,1),(self.output_shape[0]*self.output_shape[2]*self.output_shape[3],self.output_shape[1])), self.k),(self.output_shape[0],self.output_shape[2],self.output_shape[3],self.output_shape[1])).dimshuffle(0,3,1,2) * alpha / across + 1
        #lpart = conv2d(osqr, self.k, filter_shape = kernel.shape, image_shape=input.output_shape)*alpha / across + 1
        if beta!=1.0:
            lpart = lpart ** beta
        self.output = input.output / lpart

class LRN_AcrossMap1D(Layer,VisSamerank):
    
    def __init__(self, input, across, alpha = 1.0, beta = 1.0):
        assert across%2==1
        Layer.linkstruct[input].append(self)
        self.output_shape = input.output_shape
        kernel = np.zeros((self.output_shape[1],self.output_shape[1]),theano.config.floatX)
        for i in range(self.output_shape[1]):
            kernel[max(0,i-across/2):min(self.output_shape[1],i+across/2+1),i]=1.0
        self.k = theano.shared(value=kernel,name="LRN_kern_%s_%s"%(self.output_shape[1],across))
        
        osqr = input.output*input.output
        lpart = T.reshape(T.dot(T.reshape(osqr.dimshuffle(0,2,1),(self.output_shape[0]*self.output_shape[2],self.output_shape[1])), self.k),(self.output_shape[0],self.output_shape[2],self.output_shape[1])).dimshuffle(0,2,1) * alpha / across + 1
        #lpart = conv2d(osqr, self.k, filter_shape = kernel.shape, image_shape=input.output_shape)*alpha / across + 1
        if beta!=1.0:
            lpart = lpart ** beta
        self.output = input.output / lpart

class LRN_InMap(Layer, VisSamerank):
    
    def __init__(self, input, arange, alpha = 1.0, beta = 1.0):
        assert arange%2==1
        Layer.linkstruct[input].append(self)
        self.output_shape = input.output_shape
        self.k = CachedInst('theano.shared(value=np.ones((1,1,%s,%s),theano.config.floatX))'%(arange,arange))
        flatshape = (input.output_shape[0]*input.output_shape[1],1,input.output_shape[2],input.output_shape[3])
        self.flatinput = T.reshape(input.output**2,flatshape)

        conv_out = conv2d(self.flatinput, self.k, filter_shape=(1,1,arange,arange), image_shape=flatshape, border_mode='same')
        conv_out = 1+conv_out * alpha / (arange*arange)
        if beta!=1.0:
            conv_out = conv_out ** beta
        lpart = T.reshape(conv_out, input.output_shape)
        self.output = input.output / lpart

class LRN_VarianceInMap(Layer, VisSamerank):

    def __init__(self, input, arange, alpha = 1.0, beta = 1.0):
        assert arange%2==1
        Layer.linkstruct[input].append(self)
        self.output_shape = input.output_shape
        self.k = CachedInst('theano.shared(value=np.ones((1,1,%s,%s),theano.config.floatX))'%(arange,arange))
        from scipy.signal import sepfir2d
        divplain = np.ones((input.output_shape[2],input.output_shape[3]),theano.config.floatX)
        divcount = 1.0/sepfir2d(divplain,np.ones(arange,thano.config.floatX),np.ones(arange,theano.config.floatX)).reshape((1,input.output_shape[2],input.output_shape[3]))
        self.divc = theano.shared(value = divcount, name = "LRN_divc_%s_%s_%s"%(arange,input.output_shape[2],input.output_shape[3]))

        flatshape = (input.output_shape[0]*input.output_shape[1],1,input.output_shape[2],input.output_shape[3])
        self.flatinput = T.reshape(input.output,flatshape)
        conv_out = conv2d(self.flatinput, self.k, filter_shape=(1,1,arange,arange), image_shape=flatshape, border_mode='same')
        if arange!=1:
            conv_out = conv_out[:,:,arange/2:-arange/2,arange/2:-arange/2]
        mval = conv_out * self.divc
        self.flatinput = self.flatinput*self.flatinput
        conv_out = conv2d(self.flatinput, self.k, filter_shape=(1,1,arange,arange), image_shape=flatshape, border_mode='same')
        if arange!=1:
            conv_out = conv_out[:,:,arange/2:-arange/2,arange/2:-arange/2]
        conv_out = conv_out - mval
        conv_out = 1+conv_out * alpha / (arange*arange)
        if beta!=1.0:
            conv_out = conv_out ** beta
        lpart = T.reshape(conv_out, input.output_shape)
        self.output = input.output / lpart        

class Maxpool2DLayer(Layer,VisSamerank):

    def __init__(self, input, max_pool_size = (2,2), ignore_border = False, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input

        self.output = downsample.max_pool_2d(self.input, max_pool_size, ignore_border)
        if not ignore_border:
            self.output_shape = (image_shape[0], image_shape[1], (image_shape[2]+max_pool_size[0]-1)/max_pool_size[0], (image_shape[3]+max_pool_size[1]-1)/max_pool_size[1])
        else:
            self.output_shape = (image_shape[0], image_shape[1], image_shape[2]/max_pool_size[0], image_shape[3]/max_pool_size[1])

class Maxpool2D1DLayer(Layer,VisSamerank):

    def __init__(self, input, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        
        inputflat = T.reshape(self.input,(image_shape[0],image_shape[1],image_shape[2]*image_shape[3]))
        self.output = T.max(inputflat, axis=2)
        self.output_shape = (image_shape[0],image_shape[1])

class Meanpool2D1DLayer(Layer,VisSamerank):

    def __init__(self, input, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        
        inputflat = T.reshape(self.input,(image_shape[0],image_shape[1],image_shape[2]*image_shape[3]))
        self.output = T.mean(inputflat, axis=2)
        self.output_shape = (image_shape[0],image_shape[1])

class Reshape2D1DLayer(Layer, VisSamerank):

    def __init__(self, input, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        
        self.output = T.reshape(self.input,(image_shape[0],image_shape[1]*image_shape[2]*image_shape[3]))
        self.output_shape = (image_shape[0],image_shape[1]*image_shape[2]*image_shape[3])

class FlatSoftmaxLayer(Layer, VisLayer, VisSamerank):

    def __init__(self,input):
        self.output_shape = input.output_shape
        Layer.linkstruct[input].append(self)
        assert len(input.output_shape)==2
        self.output = nnet.softmax(input.output)

class SoftmaxLayer(Layer, VisLayer, VisSamerank):

    def __init__(self,input):
        self.output_shape = input.output_shape
        Layer.linkstruct[input].append(self)
        assert len(input.output_shape)==4

        x = input.output
        e_x = T.exp(x - x.max(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)
        
        self.output = out

class FullConnectLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, hidden_size, Nonlinear = True, reshape = None, input_shape = None, inc = [0], shareLayer = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if input_shape==None:
                input_shape = input.output_shape
        else:
            self.input = input

        fan_in = input_shape[1]
        filter_shape = (input_shape[1], hidden_size)

        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.uniform(
                  low=-np.sqrt(0.5/fan_in),
                  high=np.sqrt(0.5/fan_in),
                  size=filter_shape), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='Wflat_%s'%inc[0])

            b_values = np.zeros((hidden_size,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='bflat_%s'%inc[0])
        
        self.output = T.dot(self.input, self.W) + self.b.dimshuffle('x', 0)
        self.output = nonlinear(self.output, Nonlinear)
        if reshape:
            self.output = T.reshape(self.output, reshape)
            self.output_shape = reshape
        else:
            self.output_shape = (input_shape[0], hidden_size)
        self.reshape = reshape

        if shareLayer == None:
            self.params = [self.W, self.b]
        else:
            self.params = []
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
        if len(respshape)==4:
            self.output = T.tensor4('data')
            self.output_shape = (batch_size, datashape[1], datashape[2], datashape[3])
        else:
            self.output = T.matrix('data')
            self.output_shape = (batch_size, datashape[1])
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
    
class SymbolLayer(Layer, VisLayer):
    def __init__(self,input,input_shape,input_link=None):
        if input_link!=None:
            Layer.linkstruct[input_link].append(self)
        self.output = input
        self.output_shape = input_shape

class PassLayer(Layer):
    def __init__(self,input,*args,**kwargs):
        Layer.linkstruct[input].append(self)
        self.output = input.output
        self.output_shape = input.output_shape

class MaskedHengeLoss(Layer, VisLayer, LossLayer):

    def __init__(self,input,response):
        Layer.linkstruct[input].append(self)
        targets = response.resp
        mask = T.sgn(targets)
        antargets=T.switch(T.gt(targets,0),targets,1+targets)
        self.loss = self.hengeloss = T.sum((mask*(antargets-input.output)).clip(0,1e10))
        self.output = response.resp
        self.output_shape = response.resp_shape

class CrossEntropyLoss(Layer, VisLayer, LossLayer):

    def __init__(self,input,response,mask=None):
        Layer.linkstruct[input].append(self)
        targets = response.resp
        output = nnet.binary_crossentropy(input.output,targets)*(1 if mask==None else mask)
        self.loss = output.mean()
        self.output = (input.output - targets)*(1 if mask==None else mask)
        self.output_shape = response.resp_shape

class BlurSquareLoss(Layer, VisLayer, LossLayer):
    def __init__(self,input,response,blurkernel,inputshape=None,mask=None):
        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if inputshape==None: inputshape = input.output_shape
        else:
            self.input = input
        blurkernel = blurkernel.reshape((1,1)+blurkernel.shape).astype('f')
        self.kernel = theano.shared(blurkernel, name='LOSSKERNEL')
        targets = response.resp
        flattargets = targets.reshape((response.resp_shape[0]*response.resp_shape[1],1,response.resp_shape[2],response.resp_shape[3]))
        self.blurtargets = conv2d(flattargets, self.kernel, (response.resp_shape[0]*response.resp_shape[1],1,response.resp_shape[2],response.resp_shape[3]), blurkernel.shape, 'full')
        flatinput = self.input.reshape((inputshape[0]*inputshape[1],1,inputshape[2],inputshape[3]))
        self.blurinput = conv2d(flatinput, self.kernel, (inputshape[0]*inputshape[1],1,inputshape[2],inputshape[3]), blurkernel.shape, 'full')
        diff = self.blurtargets - self.blurinput
        self.loss = self.squareloss = T.sum((diff*diff) if mask==None else (diff*diff*mask))
        self.output = targets
        self.output_shape = response.resp_shape

class SquareLoss(Layer, VisLayer, LossLayer):

    def __init__(self,input,response,mask=None):
        Layer.linkstruct[input].append(self)
        targets = response.resp
        output = (targets-input.output)*(1 if mask==None else mask)
        self.loss = self.squareloss = T.sum(output*output)
        self.output_shape = response.resp_shape
        self.output = output

class CapBinarySquareLoss(Layer, VisLayer, LossLayer):

    def __init__(self,input,response,mask=None):
        Layer.linkstruct[input].append(self)
        total = T.switch(T.gt(response.resp,0), 1-input.output, input.output).clip(0, 1e10) * (1 if mask==None else mask)
        self.loss = T.sum(total**2)
        self.output = T.concatenate([total * 2 * (response.resp - 0.5), input.output, response.resp], axis=1)
        self.output_shape = input.output_shape

class SSIMLoss(Layer, VisLayer, LossLayer):
    
    def __init__(self, input, response, image_shape = None, gkern = 5, gsigma = 1.5, c1 = 6.5025, c2 = 58.5225):
        if isinstance(input, Layer):
            self.input = input.output 
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        Layer.linkstruct[input].append(self)
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

class DropOut(Layer, VisSamerank):

    def __init__(self,input,rnd,symboldropout=1):
        self.data=input.output
        Layer.linkstruct[input].append(self)
        self.output_shape=input.output_shape
        self.rnd=rnd.binomial(size=input.output.shape, n=1, p=0.5, dtype='float32')
        self.output=self.data*(1+symboldropout*(self.rnd*2-1))

class LayerbasedDropOut(Layer, VisSamerank):

    def __init__(self,input,rnd,symboldropout=1):
        self.data=input.output
        Layer.linkstruct[input].append(self)
        self.output_shape=input.output_shape
        self.rnd=rnd.binomial(size=input.output.shape[0:2], dtype='float32', ndim=2)
        self.rnd = T.shape_padright(self.rnd, len(input.output_shape)-2)
        self.output = self.data*(1+symboldropout*(self.rnd*2-1))

class LayerbasedDropOut1D(Layer, VisSamerank):

    def __init__(self,input,rnd,symboldropout=1):
        self.data=input.output
        Layer.linkstruct[input].append(self)
        self.output_shape=input.output_shape
        self.rnd=rnd.binomial(size=(input.output.shape[1],), dtype='float32')
        self.rnd = T.shape_padleft(T.shape_padright(self.rnd, len(input.output_shape)-2),1)
        self.output = self.data*(1+symboldropout*(self.rnd*2-1))

class LogSoftmaxLayer(Layer, VisLayer, VisSamerank):

    def __init__(self,input):
        self.output_shape = input.output_shape
        Layer.linkstruct[input].append(self)
        tdat = input.output.reshape((np.prod(self.output_shape[0:-1]),self.output_shape[-1]))
        tdat = tdat - tdat.max(axis=1,keepdims=True)
        tdat = tdat - T.log(T.exp(tdat).sum(axis=1,keepdims=True))

        self.output = tdat.reshape(self.output_shape)

class LabelLoss(Layer, VisLayer, LossLayer):

    def __init__(self,input,truth):
        Layer.linkstruct[input].append(self)
        self.loss = -(input.output * truth.resp).sum()
        self.output = truth.resp
        self.output_shape = truth.output_shape

def conv1d(data,kern,border_mode='valid',image_shape=None,filter_shape=None):
    return conv2d(data.reshape((data.shape[0],data.shape[1],1,data.shape[2])),kern.reshape((kern.shape[0],kern.shape[1],1,kern.shape[2])),image_shape,filter_shape,border_mode)[:,:,0,:]

class Conv1DLayer(Layer,Param,VisLayer):

    def __init__(self, rng, input, filter_shape, image_shape = None, isShrink = True, Nonlinear = "tanh", zeroone = False, inc=[0], shareLayer = None):
        if isinstance(input, Layer):
            self.input = input.output 
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        assert image_shape[1] == filter_shape[1]
        fan_in = np.prod(filter_shape[1:])
        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.uniform(
                  low=-np.sqrt(0.01/fan_in),
                  high=np.sqrt(0.01/fan_in),
                  size=filter_shape), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])
        conv_out = conv1d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="valid" if isShrink else "full")
        self.output = nonlinear(conv_out + self.b.dimshuffle('x', 0, 'x'), Nonlinear)
        if zeroone:
            self.output = (self.output+1) * 0.5

        self.output_shape = (image_shape[0], filter_shape[0],
                image_shape[2]-filter_shape[2]+1 if isShrink else image_shape[2]+filter_shape[2]-1)
        
        if shareLayer==None:
            self.params = [self.W, self.b]
        else:
            self.params = []

        inc[0] = inc[0]+1

class Maxpool1DLayer(Layer,VisSamerank):

    def __init__(self, input, max_pool_size = 2, max_pool_stride = None, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        if max_pool_stride==None: max_pool_stride = max_pool_size

        def last_pool(im_shp, p_shp, p_strd):
            rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
            assert p_strd * rval + p_shp >= im_shp
            assert p_strd * (rval - 1) + p_shp < im_shp
            return rval
        # Compute starting row of the last pool
        last_pool_r = last_pool(image_shape[2],
                                max_pool_size,
                                max_pool_stride) * max_pool_stride
        # Compute number of rows needed in image for all indexes to work out
        required_r = last_pool_r + max_pool_size
        #print last_pool_r,max_pool_size

        wide_infinity = T.alloc(dtypeX(-np.inf),
                                image_shape[0],
                                image_shape[1],
                                required_r)
        winput = T.set_subtensor(wide_infinity[:,:,0:image_shape[2]], self.input)
        mx = None
        for row_within_pool in xrange(max_pool_size):
            row_stop = last_pool_r + row_within_pool + 1
            #print row_within_pool,row_stop,max_pool_stride
            cur = winput[:,
                       :,
                       row_within_pool:row_stop:max_pool_stride]
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)

        self.output = mx
        self.output_shape = (image_shape[0], image_shape[1], (last_pool_r + max_pool_stride)/max_pool_stride)

class Reshape1D1DLayer(Layer, VisSamerank, VisLayer):

    def __init__(self, input, image_shape = None):

        if isinstance(input, Layer):
            self.input = input.output
            Layer.linkstruct[input].append(self)
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        
        self.output = T.reshape(self.input,(image_shape[0],image_shape[1]*image_shape[2]))
        self.output_shape = (image_shape[0],image_shape[1]*image_shape[2])

class Aggregation1DLayer(Layer):

    def __init__(self, *layers):

        channels = 0
        for i in layers:
            assert isinstance(i, Layer)
            channels += i.output_shape[1]

        self.output_shape = layers[0].output_shape[0], channels, layers[0].output_shape[2]
        self.output = CachedAlloc(dtypeX(0.0), *self.output_shape)
        channels = 0
        for i in layers:
            Layer.linkstruct[i].append(self)
            self.output = T.set_subtensor(self.output[:,channels:channels+i.output_shape[1]], i.output)
            channels += i.output_shape[1]

class AggregationLayer(Layer):

    def __init__(self, *layers):

        channels = 0
        for i in layers:
            assert isinstance(i, Layer)
            channels += i.output_shape[1]

        self.output_shape = (layers[0].output_shape[0], channels) + tuple(layers[0].output_shape[2:])
        self.output = T.concatenate([i.output for i in layers], axis=1)
        for i in layers:
            Layer.linkstruct[i].append(self)
Aggregation1DLayer = AggregationLayer

class Symbol1DDataLayer(Layer, VisLayer):

    def __init__(self, datashape, respshape = None, batch_size = None):
        self.output = T.tensor3('data')
        if batch_size == None:
            batch_size = respshape
            respshape = None
        self.output_shape = (batch_size, datashape[1], datashape[2])
        if respshape != None:
            if len(respshape)==3:
                self.resp = T.tensor3('resp')
                self.resp_shape = (batch_size, respshape[1], respshape[2])
            else:
                self.resp = T.matrix('resp')
                self.resp_shape = (batch_size, respshape[1])
            self.label = self.resp
        self.n_batches = datashape[0] / batch_size
        self.batch_size = batch_size
        self.data = self.output

def makesoftmaxlabel(raw, threshold = 50):
    processed = np.zeros((raw.shape[0]*raw.shape[2]*raw.shape[3],raw.shape[1]+1),'f')
    rawflat = raw.transpose((0,2,3,1)).reshape((-1,raw.shape[1]))
    labels = np.argmax(rawflat,axis=1)
    labels = np.where(rawflat[(np.arange(processed.shape[0]),labels)]>threshold,labels,raw.shape[1])
    processed[(np.arange(processed.shape[0]),labels)]=1
    processed = processed.reshape((raw.shape[0],raw.shape[2],raw.shape[3],raw.shape[1]+1)).transpose(0,3,1,2)

    return processed

def makehenge(raw, minv = 0, midv = 127, maxv = 255):
    raw = raw.astype('f')
    v = np.where(raw<midv, (raw-midv)/(midv-minv), (raw-midv)/(maxv-midv))
    return v.clip(-1,1)

class Model:

    def __init__(self, *layers):
        self.layers = layers
        for i in self.layers:
            if isinstance(i,LossLayer):
                self.loss = i
                break

    def save(self, fileobj):
        idx = 0
        params = {}
        for i in self.layers:
            idx += 1
            if isinstance(i,Param):
                for j in i.params:
                    params['%s_%s'%(idx,j.name)] = j.get_value()
            if isinstance(i,NParam):
                for j in i.nparams:
                    params['%s_NP'%idx] = j
        np.savez_compressed(fileobj, **params)

    def load(self, fileobj):
        obj = np.load(fileobj)
        idx = 0
        for i in self.layers:
            idx += 1
            if isinstance(i,Param):
                for j in i.params:
                    try:
                        assert j.get_value().shape == obj['%s_%s'%(idx,j.name)].shape
                        d = j.get_value()
                        d[:]=obj['%s_%s'%(idx,j.name)]
                        j.set_value(d)
                    except:
                        print "CANNOT SET",'%s_%s'%(idx,j.name)
            if isinstance(i,NParam):
                for j in i.nparams:
                    try:
                        j[:] = obj['%s_NP'%idx]
                    except Exception,e:
                        print e
                        print "CANNOT SET",'%s_NP'%idx
    
    def outputs(self):
        p = []
        for i in self.layers:
            if isinstance(i,VisLayer):
                if not hasattr(i,'omitvis'):
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
        for i in self.layers:
            if isinstance(i,CMomentum):
                q.extend(i.get_momentums())
            elif isinstance(i,Param):
                for j in i.params:
                    init = np.zeros_like(j.get_value())
                    q.append(theano.shared(init, name=j.name+'_momentum'))
        return q

    def loss(self):
        return self.losslayer().loss

    def losslayer(self):
        for i in self.layers:
            if isinstance(i, LossLayer):
                return i
        return None

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

try: from theano.sandbox.cuda.dnn import dnn_pool
except:
    from theano.tensor.signal.downsample import max_pool_2d
    dnn_pool = lambda a,b,c: max_pool_2d(a,b,st=c)
class SimpleShrinkshapeFractal(Layer):
    def __init__(self,inp):
        self.output = dnn_pool(inp.output,(2,2),(2,2))
        self.output_shape = inp.output_shape[0], inp.output_shape[1], inp.output_shape[2]/2, inp.output_shape[3]/2
class SimpleExpandshapeFractal(Layer):
    def __init__(self,inp,*args,**kargs):
        self.output_shape = inp.output_shape[0], inp.output_shape[1], inp.output_shape[2]*2, inp.output_shape[3]*2
        self.output = T.tile(inp.output.dimshuffle(0,1,2,'x',3,'x'),(1,1,1,2,1,2)).reshape(self.output_shape)



