import layerbase
import theano
import theano.tensor as T
import numpy as np
class learningspeed:
    def __init__(self,layertarget = 1.0, layerstr = 0.1, baserate = 1.0, basedynamic = 1e-3):
        #self.cellrate,self.cellstr = cellrate, cellstr
        self.layertarget,self.layerstr = layertarget, layerstr
        self.baserate = baserate
        self.basedynamic = basedynamic

    def fixspeed(self,model,momentums):
        paramlayers = model.paramlayers()
        coeff = []
        outs = []
        pid = 0
        for paramlayer in paramlayers:
            #For W
            D = 0
            if isinstance(paramlayer,(layerbase.ConvLayer,layerbase.ConvMaxoutLayer,layerbase.ConvKeepLayer)):
                layershape = paramlayer.params[0].get_value().shape
                fan_in = layershape[1]*layershape[2]*layershape[3]
                fan_out = np.prod(layershape)
                D = 1
            elif isinstance(paramlayer,(layerbase.FullConnectLayer)):
                layershape = paramlayer.params[0].get_value().shape
                fan_in = layershape[0]
                fan_out = np.prod(layershape)
                D = 1
            else:
                coeff.append(1)
            if D:
                layerrate = (self.layertarget*fan_out/fan_in) / (T.sum(abs(momentums[pid]))+1e-10) * self.layerstr + self.baserate * (1-self.layerstr) * self.basedynamic
                coeff.append(layerrate)
                outs.append(momentums[pid]*layerrate)
                pid+=1
            #For other
            for i in paramlayer.params[D:]:
                outs.append(momentums[pid]*self.baserate)
                pid+=1
        return outs,T.stacklists(coeff)

if __name__=="__main__":
    pass

