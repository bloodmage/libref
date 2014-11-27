import layerbase
import fractallayer_nosquare as flayer

def RepeatNetworkBuilder(inp,build,repeat,*args,**kwargs):
    layers = []
    for it in range(repeat):
        thislayers, outs = build(inp,*args,**kwargs)
        layers.extend(thislayers)
        inp = outs
    return layers, outs

def HIMBuilder(low,mid,high,midtmpl):
    aggregate = [mid]
    layers = []
    if low!=None:
        midl = flayer.ShrinkshapeFractal(low)
        aggregate.append(midl)
        layers.append(midl)
    if high!=None:
        midh = flayer.ExpandshapeFractal(high, midtmpl)
        aggregate.append(midh)
        layers.append(midh)
    agg = flayer.AggregationLayer(*aggregate)
    layers.append(agg)
    return layers,agg

ncollect = None
nstack = []
def LPush():
    global ncollect,nstack
    nstack.append(ncollect)
    ncollect = []
def LCollect(layers,func=None):
    global ncollect
    if isinstance(layers,(list,tuple)):
        ncollect.extend(layers)
        return func
    else:
        ncollect.append(layers)
        return layers
def LPop():
    global ncollect,nstack
    t = ncollect
    ncollect = nstack.pop()
    return t

__doc__ = """
Benefit writing multi-layer recurrent CNN with interlinking
For example, a 4-time 5-scale recurrent HIM CNN can be written:
====
def makenet(inp):
    REPEATS = 3
    SCALES = 5
    channels = [16,16,24,32,40]
    slayers = [inp]
    RECEPTIVE = 5
    LPush()
    #INPUT RESCALE
    for i in range(SCALES):
        inp = LCollect(ShrinkshapeMeanFractal(inp))
        slayers.append(inp)
    shapetmpl = list(slayer)
    #FIRST CONV
    slayers = [LCollect(ConvKeepLayer(rng, slayers[i], (channels[i], slayers[i].output_shape[1], RECEPTIVE,RECEPTIVE))) for i in range(SCALES)]
    #RECURRENT
    share = [None]*SCALES
    def buildrecur(inp,share):
        LPush()
        extinp = [None]+inp+[None}
        outs = []
        for i in range(SCALES):
            t=LCollect(*HIMBuilder(extinp[i],extinp[i+1],extinp[i+2],shapetmpl[i+1] if len(shapetmpl)<=i+1 else None))
            outs.append(LCollect(ConvKeepLayer(rng, t, (channels[i], t.output_shape[1], RECEPTIVE,RECEPTIVE), shareLayer = share[i], through=None if share[i]==None else 0)))
        if share[0]==None: share[:] = outs
        return LPop(), outs
    slayers = LCollect(*RepeatNetworkBuilder(slayers, buildrecur, REPEATS, share))
    #MERGE
    for i in range(SCALES-1,0,-1):
        t=LCollect(*HIMBuilder(None,slayers[i-1],slayers[i],sharetmpl[i]))
        slayers[i-1] = ConvKeepLayer(rng, t, (channels[i-1], t.output_shape[1], RECEPTIVE,RECEPTIVE))
    #OUTPUT
    outlayer = LCollect(ConvKeepLayer(rng, slayers[0], (inp.resp_shape[1], RECEPTIVE,RECEPTIVE), Nonlinear = False))
    return LPop(), outlayer
====
"""

def trainroutine(train,vis,model,savename,vispath):
    pass
