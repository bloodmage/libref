import theano
import numpy as np
import theano.tensor as T
from layerbase import Layer, Param, SymbolLayer, CMomentum
from fractallayer import dtypeX
from networkbuilder import LPush, LCollect, LPop

class LSTM(Layer, Param):
    #Assume used in scan
    def __init__(self, rng, input, linstate, loustate, szhidden, shareLayer = None, outf = T.tanh, noot = False, inc = [0]):
        filter_shape = (input.output_shape[1] + szhidden*2, szhidden*4)
        fan_in = np.sqrt(filter_shape[0]*2)
        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
        else:
            W_values = np.asarray(rng.standard_normal(filter_shape)/fan_in, dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

            b_values = np.zeros((filter_shape[1],), dtype=theano.config.floatX)
            b_values[szhidden:2*szhidden] -= 1
            b_values[:szhidden] += 1
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])
        
        combined_input = T.concatenate([input.output, linstate, loustate], axis=1)
        internal = T.dot(combined_input, self.W) + self.b.dimshuffle('x',0)
        
        in_gate = T.nnet.sigmoid(internal[:,:szhidden])
        forget_gate = T.nnet.sigmoid(internal[:,szhidden:2*szhidden])
        in_gate2 = outf(internal[:,2*szhidden:3*szhidden])
        out_gate = T.nnet.sigmoid(internal[:,3*szhidden:])
        self.hidden = forget_gate*linstate + in_gate2*in_gate
        if noot:
            self.output = outf(self.hidden)
        else:
            self.output = out_gate * outf(self.hidden)
        self.output_shape = (input.output_shape[0], szhidden)
        if shareLayer==None:
            self.params = [self.W, self.b]
            inc[0] = inc[0]+1
        else:
            self.params = []


class GLSTM(Layer, Param):
    #Assume used in scan
    def __init__(self, rng, input, linstate, loustate, szgate, szhidden, shareLayer = None, outf = T.tanh, noot = False, inc = [0]):
        filter_shape = (szgate*2, szhidden*4)
        gate_shape = (input.output_shape[1] + szhidden*2, szgate*4)

        fan_in = np.sqrt(filter_shape[0]*2)
        if shareLayer!=None:
            self.W = shareLayer.W
            self.b = shareLayer.b
            self.G = shareLayer.G
            self.gb = shareLayer.gb
        else:
            W_values = np.asarray(rng.standard_normal(filter_shape)/fan_in, dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W_%s'%inc[0])
            G_values = np.asarray(rng.standard_normal(gate_shape)/fan_in, dtype=theano.config.floatX)
            #G_values[:,szgate*2:] = 0
            self.G = theano.shared(value=G_values, name='G_%s'%inc[0])
            
            b_values = np.zeros((filter_shape[1],), dtype=theano.config.floatX)
            b_values[szhidden:2*szhidden] -= 1
            b_values[:szhidden] += 1
            self.b = theano.shared(value=b_values, name='b_%s'%inc[0])
            gb_values = np.zeros((gate_shape[1],), dtype=theano.config.floatX)
            #gb_values[szgate*2:] = 1
            self.gb = theano.shared(value=gb_values, name='gb_%s'%inc[0])
        
        combined_input = T.concatenate([input.output, linstate, loustate], axis=1)
        gating = T.tanh(T.dot(combined_input, self.G) + self.gb.dimshuffle('x',0))
        cgating = gating[:,:szgate*2]*gating[:,szgate*2:]

        internal = T.dot(cgating, self.W) + self.b.dimshuffle('x',0)
        
        in_gate = T.nnet.sigmoid(internal[:,:szhidden])
        forget_gate = T.nnet.sigmoid(internal[:,szhidden:2*szhidden])
        in_gate2 = outf(internal[:,2*szhidden:3*szhidden])
        out_gate = T.nnet.sigmoid(internal[:,3*szhidden:])
        self.hidden = forget_gate*linstate + in_gate2*in_gate
        if noot:
            self.output = outf(self.hidden)
        else:
            self.output = out_gate * outf(self.hidden)
        self.output_shape = (input.output_shape[0], szhidden)
        if shareLayer==None:
            self.params = [self.W, self.G, self.b, self.gb]
        else:
            self.params = []

        inc[0] = inc[0]+1

def LSTMScanArrayToArray(rng, inlayer, szhidden, outf = T.tanh, backwards = False):
    def oneStep(inp, laststate, lastout):
        inl = SymbolLayer(inp, (1,inlayer.output_shape[1]))
        lstmout = LCollect(LSTM(rng, inl, laststate, lastout, szhidden, outf = outf))
        return lstmout.hidden, lstmout.output
    LPush()
    firsthidden = T.alloc(dtypeX(0), 1,szhidden)
    firstout = T.alloc(dtypeX(0), 1,szhidden)
    (hiddens, outs), updates = theano.scan(fn=oneStep, outputs_info = [firsthidden, firstout], sequences=inlayer.output, go_backwards=backwards)
    lstml = LPop()[0]
    return SymbolLayer(outs.reshape((inlayer.output_shape[0], szhidden)), (inlayer.output_shape[0], szhidden)), lstml

extraHid = None
def ExtractHid():
    return extraHid

def BlockLSTMScanArrayToArray(rng, inlayer, szhidden, blocksize = 10, warmup = 10, outf = T.tanh, noot = False, backwards = False, shareLayer = None, warmupHidden = None, warmupOut = None):
    if backwards:
        inout = inlayer.output[::-1]
    else:
        inout = inlayer.output
    
    if warmupHidden!=None:
        if backwards:
            whid = warmupHidden.output[::-1]
        else:
            whid = warmupHidden.output

    if warmupOut!=None:
        if backwards:
            wout = warmupOut.output[::-1]
        else:
            wout = warmupOut.output

    #PrepareData
    totblks = (inlayer.output.shape[0]+blocksize-1) / blocksize
    def oneStep(inp, laststate, lastout):
        inl = SymbolLayer(inp, (totblks,inlayer.output_shape[1]))
        lstmout = LCollect(LSTM(rng, inl, laststate, lastout, szhidden, outf = outf, noot = noot, shareLayer = shareLayer))
        return lstmout.hidden, lstmout.output

    stackinp = T.alloc(dtypeX(0), totblks, blocksize+warmup, inlayer.output_shape[1])

    #Fill block data
    stackinp = T.set_subtensor(stackinp[:-1,warmup:],inout[:(totblks-1)*blocksize].reshape((totblks-1, blocksize, inlayer.output.shape[1])))
    stackinp = T.set_subtensor(stackinp[-1,warmup:warmup+inlayer.output.shape[0]-(totblks-1)*blocksize],inout[(totblks-1)*blocksize:].reshape((inlayer.output.shape[0]-(totblks-1)*blocksize, inlayer.output.shape[1])))
    #Fill block warmup data
    stackinp = T.set_subtensor(stackinp[1:,:warmup],stackinp[:-1,-warmup:])
    stackinp = stackinp.dimshuffle(1,0,2)
    LPush()
    #A large number
    firsthidden = T.alloc(dtypeX(0), totblks, szhidden)#T.as_tensor_variable(np.zeros((1000,szhidden),'f'))[:totblks]
    if warmupHidden:
        firsthidden = T.set_subtensor(firsthidden[warmup/blocksize+1:], whid[-warmup+blocksize*(warmup/blocksize+1):-warmup+blocksize*totblks:blocksize])
    firstout = T.alloc(dtypeX(0), totblks, szhidden)#T.as_tensor_variable(np.zeros((1000,szhidden),'f'))[:totblks]
    if warmupOut:
        firstout = T.set_subtensor(firstout[warmup/blocksize+1:], wout[-warmup+blocksize*(warmup/blocksize+1):-warmup+blocksize*totblks:blocksize])
    (hiddens, outs), updates = theano.scan(fn=oneStep, outputs_info = [firsthidden, firstout], sequences=stackinp)
    lstml = LPop()[0]
    #ExpandData
    hiddens = hiddens.dimshuffle(1,0,2)
    hiddens = hiddens[:,warmup:].reshape((totblks*blocksize,szhidden))[:inlayer.output.shape[0]]
    outs = outs.dimshuffle(1,0,2)
    outs = outs[:,warmup:].reshape((totblks*blocksize,szhidden))[:inlayer.output.shape[0]]
    if backwards:
        hiddens = hiddens[::-1]
        outs = outs[::-1]
    global extraHid
    extraHid = SymbolLayer(hiddens, (inlayer.output_shape[0], szhidden))
    return SymbolLayer(outs, (inlayer.output_shape[0], szhidden)), lstml

def BlockLSTMUnrollArrayToArray(rng, inlayer, szhidden, blocksize = 10, warmup = 10, outf = T.tanh, noot = False, backwards = False, shareLayer = None, warmupHidden = None, warmupOut = None):
    if backwards:
        inout = inlayer.output[::-1]
    else:
        inout = inlayer.output
    
    if warmupHidden!=None:
        if backwards:
            whid = warmupHidden.output[::-1]
        else:
            whid = warmupHidden.output

    if warmupOut!=None:
        if backwards:
            wout = warmupOut.output[::-1]
        else:
            wout = warmupOut.output

    #PrepareData
    totblks = (inlayer.output.shape[0]+blocksize-1) / blocksize
    def oneStep(inp, laststate, lastout):
        inl = SymbolLayer(inp, (totblks,inlayer.output_shape[1]))
        lstmout = LSTM(rng, inl, laststate, lastout, szhidden, outf = outf, noot = noot, shareLayer = shareLayer)
        return lstmout.hidden, lstmout.output, lstmout

    stackinp = T.alloc(dtypeX(0), totblks, blocksize+warmup, inlayer.output_shape[1])

    #Fill block data
    stackinp = T.set_subtensor(stackinp[:-1,warmup:],inout[:(totblks-1)*blocksize].reshape((totblks-1, blocksize, inlayer.output.shape[1])))
    stackinp = T.set_subtensor(stackinp[-1,warmup:warmup+inlayer.output.shape[0]-(totblks-1)*blocksize],inout[(totblks-1)*blocksize:].reshape((inlayer.output.shape[0]-(totblks-1)*blocksize, inlayer.output.shape[1])))
    #Fill block warmup data
    stackinp = T.set_subtensor(stackinp[1:,:warmup],stackinp[:-1,-warmup:])
    stackinp = stackinp.dimshuffle(1,0,2)
    #A large number
    firsthidden = T.alloc(dtypeX(0), totblks, szhidden)#T.as_tensor_variable(np.zeros((1000,szhidden),'f'))[:totblks]
    if warmupHidden:
        firsthidden = T.set_subtensor(firsthidden[warmup/blocksize+1:], whid[-warmup+blocksize*(warmup/blocksize+1):-warmup+blocksize*totblks:blocksize])
    firstout = T.alloc(dtypeX(0), totblks, szhidden)#T.as_tensor_variable(np.zeros((1000,szhidden),'f'))[:totblks]
    if warmupOut:
        firstout = T.set_subtensor(firstout[warmup/blocksize+1:], wout[-warmup+blocksize*(warmup/blocksize+1):-warmup+blocksize*totblks:blocksize])

    hiddens = []
    outs = []
    firstshare = None
    for i in range(warmup):
        firsthidden, firstout, shareLayer = oneStep(stackinp[i], firsthidden, firstout)
        if firstshare==None: firstshare = shareLayer
    for i in range(blocksize):
        firsthidden, firstout, shareLayer = oneStep(stackinp[i+warmup], firsthidden, firstout)
        if firstshare==None: firstshare = shareLayer
        hiddens.append(firsthidden)
        outs.append(firstout)

    hiddens = T.stack(*hiddens)
    outs = T.stack(*outs)
    #ExpandData (warmup is automatically eatten)
    hiddens = hiddens.dimshuffle(1,0,2)
    hiddens = hiddens.reshape((totblks*blocksize,szhidden))[:inlayer.output.shape[0]]
    outs = outs.dimshuffle(1,0,2)
    outs = outs.reshape((totblks*blocksize,szhidden))[:inlayer.output.shape[0]]
    if backwards:
        hiddens = hiddens[::-1]
        outs = outs[::-1]
    global extraHid
    extraHid = SymbolLayer(hiddens, (inlayer.output_shape[0], szhidden))
    return SymbolLayer(outs, (inlayer.output_shape[0], szhidden)), firstshare


def BlockGLSTMScanArrayToArray(rng, inlayer, szgate, szhidden, blocksize = 10, warmup = 10, outf = T.tanh, noot = False, backwards = False, shareLayer = None, warmupHidden = None, warmupOut = None):
    if backwards:
        inout = inlayer.output[::-1]
    else:
        inout = inlayer.output
    
    if warmupHidden!=None:
        if backwards:
            whid = warmupHidden.output[::-1]
        else:
            whid = warmupHidden.output

    if warmupOut!=None:
        if backwards:
            wout = warmupOut.output[::-1]
        else:
            wout = warmupOut.output

    #PrepareData
    totblks = (inlayer.output.shape[0]+blocksize-1) / blocksize
    def oneStep(inp, laststate, lastout):
        inl = SymbolLayer(inp, (totblks,inlayer.output_shape[1]))
        lstmout = LCollect(GLSTM(rng, inl, laststate, lastout, szgate, szhidden, outf = outf, noot = noot, shareLayer = shareLayer))
        return lstmout.hidden, lstmout.output

    stackinp = T.alloc(dtypeX(0), totblks, blocksize+warmup, inlayer.output_shape[1])

    #Fill block data
    stackinp = T.set_subtensor(stackinp[:-1,warmup:],inout[:(totblks-1)*blocksize].reshape((totblks-1, blocksize, inlayer.output.shape[1])))
    stackinp = T.set_subtensor(stackinp[-1,warmup:warmup+inlayer.output.shape[0]-(totblks-1)*blocksize],inout[(totblks-1)*blocksize:].reshape((inlayer.output.shape[0]-(totblks-1)*blocksize, inlayer.output.shape[1])))
    #Fill block warmup data
    stackinp = T.set_subtensor(stackinp[1:,:warmup],stackinp[:-1,-warmup:])
    stackinp = stackinp.dimshuffle(1,0,2)
    LPush()
    #A large number
    firsthidden = T.alloc(dtypeX(0), totblks, szhidden)#T.as_tensor_variable(np.zeros((1000,szhidden),'f'))[:totblks]
    if warmupHidden:
        firsthidden = T.set_subtensor(firsthidden[warmup/blocksize+1:], whid[-warmup+blocksize*(warmup/blocksize+1):-warmup+blocksize*totblks:blocksize])
    firstout = T.alloc(dtypeX(0), totblks, szhidden)#T.as_tensor_variable(np.zeros((1000,szhidden),'f'))[:totblks]
    if warmupOut:
        firstout = T.set_subtensor(firstout[warmup/blocksize+1:], wout[-warmup+blocksize*(warmup/blocksize+1):-warmup+blocksize*totblks:blocksize])
    (hiddens, outs), updates = theano.scan(fn=oneStep, outputs_info = [firsthidden, firstout], sequences=stackinp)
    lstml = LPop()[0]
    #ExpandData
    hiddens = hiddens.dimshuffle(1,0,2)
    hiddens = hiddens[:,warmup:].reshape((totblks*blocksize,szhidden))[:inlayer.output.shape[0]]
    outs = outs.dimshuffle(1,0,2)
    outs = outs[:,warmup:].reshape((totblks*blocksize,szhidden))[:inlayer.output.shape[0]]
    if backwards:
        hiddens = hiddens[::-1]
        outs = outs[::-1]
    global extraHid
    extraHid = SymbolLayer(hiddens, (inlayer.output_shape[0], szhidden))
    return SymbolLayer(outs, (inlayer.output_shape[0], szhidden)), lstml

class ShrinkShapeFractal1D(Layer):
    def __init__(self, input):
        #A 3in1 maxpooling
        self.output_shape = input.output_shape[0]/2, input.output_shape[1]
        self.origlayer = input
        self.output = input.output[::2]
        self.output = T.set_subtensor(self.output[:input.output.shape[0]/2], T.maximum(self.output[:input.output.shape[0]/2], input.output[1::2]))
        self.output = T.set_subtensor(self.output[1:], T.maximum(self.output[1:], input.output[1:-1:2]))

class ExpandShapeFractal1D(Layer):
    def __init__(self, input, orig):
        assert isinstance(orig, ShrinkShapeFractal1D)
        self.output_shape = orig.origlayer.output_shape
        self.output = T.alloc(dtypeX(0), orig.origlayer.output.shape[0], orig.origlayer.output.shape[1])
        self.output = T.set_subtensor(self.output[::2], input.output)
        self.output = T.set_subtensor(self.output[1::2], input.output[:orig.origlayer.output.shape[0]/2])

import theano.sandbox.cuda as cuda
class GPUSharedThunkOp(theano.Op):

    def __init__(self, thunkfunc, thunkout, thunkgrad):
        self.thunkfunc = thunkfunc
        self.thunkout = thunkout
        self.thunkgrad = thunkgrad
        if self.thunkout!=None:
            self.infer_shape = lambda node, i0_shapes: [self.thunkout(i0_shapes)]
        import gc
        gc.enable()

    def make_node(self, inp, inp2 = None):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))
        assert inp.dtype == "float32"
        if inp2==None:
            return theano.Apply(self, [inp], [inp.type()])
        inp2 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp2))
        assert inp2 == None or inp2.dtype == "float32"
        return theano.Apply(self, [inp, inp2], [inp.type()])

    def perform(self, node, inputs, output_storage):
        #print 'DEBUG: PERFORM', inputs, output_storage
        #print 'SHAPE:', [i.shape for i in inputs]
        tout = self.thunkfunc(*inputs)
        #print 'OUTPUT:', tout
        #print 'SHAPE:', tout.shape
        output_storage[0][0] = tout

    def grad(self, inputs, g):
        assert len(inputs)==1
        assert len(g)==1
        return [GPUSharedThunkOp(self.thunkgrad, None, None)(inputs[0], g[0])]

def SharedThunkLSTMFunc(rst, inpshape, oupshape, noot, backwards, MOMENTUM, LEARN_RATE):
    print "MAKE INNER FUNCTION", inpshape, oupshape, noot, backwards
    cuda2d = cuda.CudaNdarrayType((False,False))#T.fmatrix
    isym = SymbolLayer(cuda2d(), (100, inpshape))
    oval, l1f = BlockLSTMUnrollArrayToArray(rst, isym, oupshape, noot = noot, backwards = backwards)
    oflag = cuda2d()

    oupfunc = theano.function([isym.output], cuda.basic_ops.as_cuda_ndarray_variable(oval.output))
    infshape = lambda x0: (x0[0][0],oupshape)
    #GRADFUNC
    g = T.sum(oval.output * oflag)
    iglist = T.grad(g, [isym.output] + l1f.params)
    olist = iglist[0]
    glist = iglist[1:]
    #Generate MOMENTUM
    mom = []
    for i in l1f.params:
        init = np.zeros_like(i.get_value())
        mom.append(theano.shared(init, name=i.name+'_momentum_ct'))
    #Additive update
    updates = []
    for i,j in zip(glist, mom):
        updates.append((j, j-i*LEARN_RATE))
    momup = []
    for i in mom:
        momup.append((i, i*MOMENTUM))
    print "MAIN UPDATES",updates
    resetmom = theano.function([],[], updates=momup)
    getgrad = theano.function([isym.output, oflag], cuda.basic_ops.as_cuda_ndarray_variable(olist), updates=updates)
    sharedop = GPUSharedThunkOp(oupfunc, infshape, getgrad)
    #Make sharedlayer
    class SharedLayer(Layer, Param, CMomentum):
        def get_momentums(self): return []
        def __init__(self, inp, paramroot = False):
            if paramroot:
                self.params = l1f.params
                self.get_momentums = lambda: mom
            else:
                self.params = []
            self.output = sharedop(inp.output)
            self.output_shape = oval.output_shape

    return SharedLayer, resetmom

