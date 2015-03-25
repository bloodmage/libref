import theano
import theano.tensor as T
from layerbase import Layer, Param, VisLayer, NParam, nonlinear, LossLayer
from fractallayer import dtypeX
import numpy as np

class RNGMiddleware:
    def RNG_GEN(self, rng, inputs, outputs):
        inc = getattr(self, 'inc_RNG', 0)
        values = np.asarray(rng.uniform(low=-np.sqrt(2.0/inputs), high=np.sqrt(2.0/inputs),
            size=(outputs, inputs)), dtype=theano.config.floatX)
        shrd = theano.shared(value=values, name='RNG%s'%inc)
        self.inc_RNG = inc+1
        return shrd

    def ZERO_GEN(self, outputs):
        inc = getattr(self, 'inc_RNG', 0)
        b_values = np.zeros((outputs,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='ZERO%s'%inc)
        self.inc_RNG = inc+1
        return b

    def ZERO_GEN_SYMBOL(self, *s):
        return T.alloc(dtypeX(0),*s)

class BidirectionalLimitedRecurrentLayer(Layer, Param, VisLayer, RNGMiddleware):
    def __init__(self, rng, originput, lastinput, hiddenl, hiddenr, stepl, stepr, outputs = None, sharedlayers = None, nlscan = 'tanh', nlout = 'tanh'):
        if hiddenr == None: hiddenr = hiddenl
        if stepr == None: stepr = stepl
        assert len(originput.output_shape)==3
        assert lastinput == None or originput.output_shape[0] == lastinput.output_shape[0]
        ilayers = originput.output_shape[1]
        if sharedlayers!=None: wd = sharedlayers.__dict__
        else: wd = {}
        self.Win_hl = Win_hl = wd.get('Win_hl') or self.RNG_GEN(rng, ilayers, hiddenl)
        self.Win_hr = Win_hr = wd.get('Win_hr') or self.RNG_GEN(rng, ilayers, hiddenr)
        self.Wprev = Wprev = wd.get('Wprev') or self.RNG_GEN(rng, hiddenl, hiddenl)
        self.Wnext = Wnext = wd.get('Wnext') or self.RNG_GEN(rng, hiddenr, hiddenr)
        self.bl = bl = wd.get('bl') or self.ZERO_GEN(hiddenl)
        self.br = br = wd.get('br') or self.ZERO_GEN(hiddenr)
        if lastinput != None:
            ilast = lastinput.output_shape[1]
            self.Wlastl = Wlastl = wd.get('Wlastl') or self.RNG_GEN(rng, ilast, hiddenl)
            self.Wlastr = Wlastr = wd.get('Wlastr') or self.RNG_GEN(rng, ilast, hiddenr)
        if outputs != None:
            self.Woutput = Woutput = wd.get('Woutput') or self.RNG_GEN(rng, hiddenl+hiddenr, outputs)
            self.boutput = boutput = wd.get('boutput') or self.ZERO_GEN(outputs)
        else:
            #A aggregation output solution
            pass
        
        vl0 = self.ZERO_GEN_SYMBOL(originput.output_shape[0], hiddenl)
        vr0 = self.ZERO_GEN_SYMBOL(originput.output_shape[0], hiddenr)

        
        #Real work
        self.odim = odim = (originput.output_shape[1],originput.output_shape[0])
        ldim = None
        self.odl = odl = (hiddenl,originput.output_shape[0])
        self.odr = odr = (hiddenr,originput.output_shape[0])
        self.T_orig_o = T_orig_o = originput.output.dimshuffle(2,0,1)
        if lastinput != None:
            self.ldim = ldim = (lastinput.output_shape[1],lastinput.output_shape[0])
            self.T_last_o = T_last_o = lastinput.output.dimshuffle(2,0,1)
            statL = T.alloc(dtypeX(0), originput.output_shape[2], originput.output_shape[0], hiddenl)
            for steps in range(stepl):
                if steps!=0:
                    statL = T.set_subtensor(statL[1:], statL[:-1])
                    statL = T.set_subtensor(statL[0], dtypeX(0))
                t = T.tensordot(statL, Wprev, [[2], [1]]) + T.tensordot(T_orig_o, Win_hl, [[2], [1]]) + T.tensordot(T_last_o, Wlastl, [[2], [1]])
                statL = nonlinear(t + bl.dimshuffle('x','x',0), nlscan)
            self.currl = currl = statL
            statR = T.alloc(dtypeX(0), originput.output_shape[2], originput.output_shape[0], hiddenr)
            for steps in range(stepr):
                if steps!=0:
                    statR = T.set_subtensor(statR[:-1], statR[1:])
                    statR = T.set_subtensor(statR[-1], dtypeX(0))
                t = T.tensordot(statR, Wnext, [[2], [1]]) + T.tensordot(T_orig_o, Win_hr, [[2], [1]]) + T.tensordot(T_last_o, Wlastr, [[2], [1]])
                statR = nonlinear(t + br.dimshuffle('x','x',0), nlscan)
            self.currr = currr = statR
        else:
            statL = T.alloc(dtypeX(0), originput.output_shape[2], originput.output_shape[0], hiddenl)
            for steps in range(stepl):
                if steps!=0:
                    statL = T.set_subtensor(statL[1:], statL[:-1])
                    statL = T.set_subtensor(statL[0], dtypeX(0))
                t = T.tensordot(statL, Wprev, [[2], [1]]) + T.tensordot(T_orig_o, Win_hl, [[2], [1]])
                statL = nonlinear(t + bl.dimshuffle('x','x',0), nlscan)
            self.currl = currl = statL
            statR = T.alloc(dtypeX(0), originput.output_shape[2], originput.output_shape[0], hiddenr)
            for steps in range(stepr):
                if steps!=0:
                    statR = T.set_subtensor(statR[:-1], statR[1:])
                    statR = T.set_subtensor(statR[-1], dtypeX(0))
                t = T.tensordot(statR, Wnext, [[2], [1]]) + T.tensordot(T_orig_o, Win_hr, [[2], [1]])
                statR = nonlinear(t + br.dimshuffle('x','x',0), nlscan)
            self.currr = currr = statR

        #Make output
        aggout = T.concatenate([currl, currr], axis=2)
        if outputs != None:
            #Transpose through another layer
            self.output = nonlinear(T.tensordot(aggout, Woutput, [[2],[1]]) + boutput.dimshuffle('x','x',0), nlout).dimshuffle(1,2,0)
            self.output_shape = [originput.output_shape[0], outputs, originput.output_shape[2]]
        else:
            self.output = aggout.dimshuffle(1,2,0)
            self.output_shape = [originput.output_shape[0], hiddenl+hiddenr, originput.output_shape[2]]
        
        self.params = [Win_hl, Win_hr, Wprev, Wnext, bl, br]
        if lastinput!=None:
            self.params.extend([Wlastl, Wlastr])
        if outputs!=None:
            self.params.extend([Woutput, boutput])

        if sharedlayers!=None:
            for i in sharedlayers.params:
                if i in self.params:
                    self.params.remove(i)

class BidirectionalRecurrentLayer(Layer, Param, VisLayer, RNGMiddleware):
    def __init__(self, rng, originput, lastinput, hiddenl, hiddenr, outputs = None, sharedlayers = None, nlscan = 'tanh', nlout = 'tanh'):
        if hiddenr == None: hiddenr = hiddenl
        assert len(originput.output_shape)==3
        assert originput.output_shape[0] == lastinput.output_shape[0]
        ilayers = originput.output_shape[1]
        if sharedlayers!=None: wd = sharedlayers.__dict__
        else: wd = {}
        self.Win_hl = Win_hl = wd.get('Win_hl') or self.RNG_GEN(rng, ilayers, hiddenl)
        self.Win_hr = Win_hr = wd.get('Win_hr') or self.RNG_GEN(rng, ilayers, hiddenr)
        self.Wprev = Wprev = wd.get('Wprev') or self.RNG_GEN(rng, hiddenl, hiddenl)
        self.Wnext = Wnext = wd.get('Wnext') or self.RNG_GEN(rng, hiddenr, hiddenr)
        self.bl = bl = wd.get('bl') or self.ZERO_GEN(hiddenl)
        self.br = br = wd.get('br') or self.ZERO_GEN(hiddenr)
        if lastinput != None:
            ilast = lastinput.output_shape[1]
            self.Wlastl = Wlastl = wd.get('Wlastl') or self.RNG_GEN(rng, ilast, hiddenl)
            self.Wlastr = Wlastr = wd.get('Wlastr') or self.RNG_GEN(rng, ilast, hiddenr)
        if outputs != None:
            self.Woutput = Woutput = wd.get('Woutput') or self.RNG_GEN(rng, hiddenl+hiddenr, outputs)
            self.boutput = boutput = wd.get('boutput') or self.ZERO_GEN(outputs)
        else:
            #A aggregation output solution
            pass
        
        vl0 = self.ZERO_GEN_SYMBOL(hiddenl*originput.output_shape[0])
        vr0 = self.ZERO_GEN_SYMBOL(hiddenr*originput.output_shape[0])

        
        #Real work
        self.odim = odim = (originput.output_shape[1],originput.output_shape[0])
        ldim = None
        self.odl = odl = (hiddenl,originput.output_shape[0])
        self.odr = odr = (hiddenr,originput.output_shape[0])
        def scanl(inorig, inlast, inprev=None):
            if inprev==None: inprev, inlast = inlast, inprev

            if inlast==None:
                t = T.dot(Wprev,inprev.reshape(odl)) + T.dot(Win_hl,inorig.reshape(odim))
            else:
                t = T.dot(Wprev,inprev.reshape(odl)) + T.dot(Win_hl,inorig.reshape(odim)) + T.dot(Wlastl,inlast.reshape(ldim))
            return nonlinear(t + bl.dimshuffle(0,'x'), nlscan).flatten()
        def scanr(inorig, inlast, innext=None):
            if innext==None: innext, inlast = inlast, innext

            if inlast==None:
                t = T.dot(Wnext,innext.reshape(odr)) + T.dot(Win_hr,inorig.reshape(odim))
            else:
                t = T.dot(Wnext,innext.reshape(odr)) + T.dot(Win_hr,inorig.reshape(odim)) + T.dot(Wlastr,inlast.reshape(ldim))
            return nonlinear(t + br.dimshuffle(0,'x'), nlscan).flatten()
        
        self.T_orig_o = T_orig_o = originput.output.dimshuffle(2,1,0).reshape((originput.output_shape[2],odim[0]*odim[1]))
        if lastinput != None:
            self.ldim = ldim = (lastinput.output_shape[1],lastinput.output_shape[0])
            self.T_last_o = T_last_o = lastinput.output.dimshuffle(2,1,0).reshape((lastinput.output_shape[2],ldim[0]*ldim[1]))
            self.currl = currl = theano.scan(scanl, outputs_info = vl0, sequences = [T_orig_o, T_last_o])[0]
            self.currr = currr = theano.scan(scanr, outputs_info = vr0, sequences = [T_orig_o, T_last_o], go_backwards = True)[0]
            currl = currl.reshape((originput.output_shape[2],)+odl)
            currr = currr.reshape((originput.output_shape[2],)+odr)
        else:
            self.currl = currl = theano.scan(scanl, outputs_info = vl0, sequences = [T_orig_o])[0]
            self.currr = currr = theano.scan(scanr, outputs_info = vr0, sequences = [T_orig_o], go_backwards = True)[0]
            currl = currl.reshape((originput.output_shape[2],)+odl)
            currr = currr.reshape((originput.output_shape[2],)+odr)

        #Make output
        aggout = T.concatenate([currl, currr], axis=1)
        if outputs != None:
            #Transpose through another layer
            self.output = nonlinear(T.tensordot(aggout, Woutput, [[1],[1]]) + boutput.dimshuffle('x','x',0), nlout).dimshuffle(1,2,0)
            self.output_shape = [originput.output_shape[0], outputs, originput.output_shape[2]]
        else:
            self.output = aggout.dimshuffle(2,1,0)
            self.output_shape = [originput.output_shape[0], hiddenl+hiddenr, originput.output_shape[2]]
        
        self.params = [Win_hl, Win_hr, Wprev, Wnext, bl, br]
        if lastinput!=None:
            self.params.extend([Wlastl, Wlastr])
        if outputs!=None:
            self.params.extend([Woutput, boutput])

        if sharedlayers!=None:
            for i in sharedlayers.params:
                if i in self.params:
                    self.params.remove(i)

class AdaptMeanPoolingLayer(Layer):
    def __init__(self, input, total, current):
        assert len(input.output_shape)==3
        #SymbolPoolsize
        self.poolsize = poolsize = T.cast(T.floor(input.output_shape[2] ** (1.0/(total-current+1))),'int32')
        self.poolresp = poolresp = T.cast((input.output_shape[2] + poolsize - 1) / poolsize,'int32')
        #Pad to normal size
        rectinput = T.alloc(dtypeX(0), input.output_shape[0], input.output_shape[1], poolsize*poolresp)
        rectinput = T.set_subtensor(rectinput[:,:,:input.output_shape[2]],input.output)
        rectinput = rectinput.reshape((input.output_shape[0], input.output_shape[1], poolresp, poolsize))
        #Reshape and mean axis out
        self.output = T.mean(rectinput,3)
        #Special deal with last index
        self.output = T.set_subtensor(self.output[:,:,-1], T.mean(rectinput[:,:,-1,:input.output_shape[2]-(poolresp-1)*poolsize],2))
        self.output_shape = (input.output_shape[0], input.output_shape[1], poolresp)

class MeanPoolingLayer(Layer):
    def __init__(self, input):
        assert len(input.output_shape)==3
        
        self.output = T.mean(input.output,2)
        self.output_shape = (input.output_shape[0], input.output_shape[1])

class SymbolParamLayer(Layer, Param, NParam):
    def __init__(self, params = (), nparams = ()):
        self.params = params
        self.nparams = nparams
        
        #A dummy output
        self.output = None
        self.output_shape = ()

class SymbolLossLayer(Layer, LossLayer):
    def __init__(self, o, os, loss):
        self.output = o
        self.output_shape = os
        self.loss = loss

class TakeLayer(Layer, Param, RNGMiddleware):
    def __init__(self, rng, indices, slots, dims, sharedlayers = None):
        
        if sharedlayers!=None:
            self.taketable = sharedlayers.taketable
        else:
            self.taketable = self.RNG_GEN(rng, dims, slots)
        flatinp = indices.output.flatten()

        flatout = self.taketable.take(flatinp, axis=0)
        self.output = flatout.reshape(indices.output_shape + (dims,)).dimshuffle(0, len(indices.output_shape), *range(1,len(indices.output_shape)))
        self.output_shape = indices.output_shape[0:1] + (dims,) + indices.output_shape[1:]
        if sharedlayers!=None:
            self.params = []
        else:
            self.params = [self.taketable]

class HuffmannLossLayer(Layer, LossLayer):
    def __init__(self, input, target, huffmannconf, extmask = 1):
        #Config to gpu
        huffmannmults = np.array([[1 if j!=' ' else 0 for j in i] for _, i in huffmannconf],'f')
        huffmannapps = np.array([[1 if j=='1' else 0 for j in i] for _, i in huffmannconf],'f')

        #Loss function: bits not correct
        features = huffmannapps.take(target.resp.flatten(), axis=0)
        masks = huffmannmults.take(target.resp.flatten(), axis=0)
        tocmp = input.output.dimshuffle(0, *range(2,len(input.output_shape)), 1).flatten()
        self.loss = T.sum((features - tocmp)**2 * masks * extmask)

        self.output = target.resp
        self.output_shape = target.resp_shape


