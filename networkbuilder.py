import layerbase
import fractallayer_nosquare as flayer
from layerbase import safefile, DrawPatch
import PIL
import PIL.Image
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



def trainroutine(ftrain,model,savename,vispath,fdatagen,fvis=None,fcheck=None,fcheckgen=None,TOLTIMES=5,BATCHSTEP=10,LEARNRATEVAR=None,LEARNRATETARGET=10.0,LEARNADJUST=1.01):

    with safefile(savename) as loadf:
        if loadf:
            model.load(loadf.rb())

    import sys
    LOSS0 = 1e100
    tol = 0
    d = 0
    if not os.path.exists(vispath):
        os.mkdir(vispath)

    while True:
        gen = fdatagen()
        d += [float(t) for t in ftrain(*gen)][1]
        sys.stdout.write('.')
        sys.stdout.flush()
        if i % BATCHSTEP == BATCHSTEP-1:
            print d,
            if LEARNRATEVAR!=None:
                lval = LEARNRATEVAR.get_value()
                if d>LEARNRATETARGET*BATCHSTEP:
                    lval /= LEARNADJUST
                else:
                    lval *= LEARNADJUST
                LEARNRATEVAR.set_value(lval)
                print lval,
            d = 0
            print "DRAW"
            #Draw model
            layer = 0
            for i in model.paramlayers():
                if len(i.params)<1: continue
                layer += 1
                param = i.params[0].get_value()
                if len(param.shape)!=4:
                    if hasattr(i,'reshape') and i.reshape!=None:
                        PIL.Image.fromarray(DrawPatch(param.reshape((-1,)+i.reshape[1:]))).save(os.path.join(vispath,'layer_%s.jpg'%layer), quality=100)
                    else:
                        PIL.Image.fromarray(np.array((param-np.min(param))/(np.max(param)-np.min(param))*255,np.uint8)).save(os.path.join(vispath,'layer_%s.png'%layer))
                    continue
                PIL.Image.fromarray(DrawPatch(param)).save(os.path.join(vispath,'layer_%s.jpg'%layer), quality=100)
            #Draw response
            if fvis!=None:
                resp = fvis(din,outf,maskf[:,0,:,:])
                layer = 0
                for i in resp:
                    layer += 1
                    if len(i.shape)!=4:
                        PIL.Image.fromarray(np.array((i-np.min(i))/(np.max(i)-np.min(i))*255,np.uint8)).save(os.path.join(vispath,'resp%s.png'%layer))
                        continue
                    PIL.Image.fromarray(DrawPatch(i[0:1])).save(os.path.join(vispath,'sfvisrecur/resp%s.jpg'%layer), quality=100)
            #Check validset
            if fcheckgen!=None:
                LOSS1 = 0.0
                for j in fcheckgen():
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    LOSS1 += valid_model(*j)
                print LOSS1
                if LOSS1>LOSS0:
                    print "Converge on validset"
                    tol+=1
                    if tol>TOLTIMES:
                        sys.exit(0)
                else:
                    tol=0
                print "NEW LOSS",LOSS1
                LOSS0 = LOSS1
        #Save model
        with safefile(savename) as savef:
            model.save(savef.wb())

def makenet(inp):
    """
    Benefit writing multi-layer recurrent CNN with interlinking
    For example, a 4-time 5-scale recurrent HIM CNN can be written as this function
    """
    from fractallayer_nosquare import ShrinkshapeMeanFractal, ConvKeepLayer
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

