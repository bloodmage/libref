import layerbase
import fractallayer_nosquare as flayer
from fractallayer import dtypeX
import new

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
        midh = flayer.ExpandshapeFractal(high, midtmpl, calibrate = False)
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

def TwoAheadProducer(prodfunc):
    import threading, Queue
    dataqueue = Queue.Queue(1)
    def producer():
        while True:
            dataqueue.put(prodfunc())
    def gen():
        return dataqueue.get()
    ahead = threading.Thread(target=producer)
    ahead.daemon=True
    ahead.start()
    return gen

def __MPWorkerFunc(invokestr,queue):
    module,func = invokestr.split(' ')
    mod = __import__(module)
    fobj = getattr(mod,func)
    while True:
        queue.put(fobj())

def __MPDrawFunc(vispath,queue):
    import PIL
    import PIL.Image
    from layerbase import DrawPatch
    import os
    while True:
        draws,resp = queue.get()
        try:
            for layer,param,reshape in draws:
                if len(param.shape)!=4:
                    if reshape!=None:
                        PIL.Image.fromarray(DrawPatch(param.reshape((-1,)+reshape[1:]))).save(os.path.join(vispath,'layer_%s.jpg'%layer), quality=100)
                    else:
                        PIL.Image.fromarray(np.array((param-np.min(param))/(np.max(param)-np.min(param))*255,np.uint8)).save(os.path.join(vispath,'layer_%s.png'%layer))
                    continue
                PIL.Image.fromarray(DrawPatch(param)).save(os.path.join(vispath,'layer_%s.jpg'%layer), quality=100)
            #Draw response
            layer = 0
            for i in resp:
                layer += 1
                if len(i.shape)!=4:
                    PIL.Image.fromarray(np.array((i-np.min(i))/(np.max(i)-np.min(i))*255,np.uint8)).save(os.path.join(vispath,'resp%s.png'%layer))
                    continue
                PIL.Image.fromarray(DrawPatch(i[0:1])).save(os.path.join(vispath,'resp%s.jpg'%layer), quality=100)
        except IOError:
            print "DRAW FAILED, IGNORE"

drawqueue = None
def MPDrawInitializer(vispath):
    import multiprocessing
    dataqueue = multiprocessing.Queue(1)
    p = multiprocessing.Process(target=__MPDrawFunc, args=(vispath,dataqueue))
    p.daemon = True
    p.start()
    global drawqueue
    drawqueue = dataqueue
def MPDrawWriter(*data):
    global drawqueue
    drawqueue.put(data)

def MPTwoAheadProducer(prodfuncstr):
    import multiprocessing
    dataqueue = multiprocessing.Queue(1)
    p = multiprocessing.Process(target=__MPWorkerFunc, args=(prodfuncstr, dataqueue))
    p.daemon = True
    p.start()
    def gen():
        return dataqueue.get()
    return gen

TRAINSETTINGS = new.classobj('settings',(),{})()

def trainroutine(ftrain,model,savename,vispath,fdatagen,fvis=None,fcheck=None,fcheckgen=None,TOLTIMES=5,BATCHSTEP=10,LEARNRATEVAR=None,LEARNRATETARGET=10.0,LEARNADJUST=1.01, remotemonitor = False, sameranks = [], longrangecheck = None, longrangeperiod = None,totalsteps = None):
    global TRAINSETTINGS
    TRAINSETTINGS.TOLTIMES = TOLTIMES
    TRAINSETTINGS.BATCHSTEP = BATCHSTEP
    TRAINSETTINGS.LEARNRATEVAR = LEARNRATEVAR
    TRAINSETTINGS.LEARNRATETARGET = LEARNRATETARGET
    TRAINSETTINGS.LEARNADJUST = LEARNADJUST
    TRAINSETTINGS.TOTALSTEPS = totalsteps
    from layerbase import safefile
    import sys, os
    if remotemonitor!=False:
        import modelrecord
        if remotemonitor==None: modrec = remotemonitor.Record()
        else: modrec = modelrecord.Record(remotemonitor)
        modrec.genmeta(model, sameranks)
    else:
        modrec = None

    with safefile(savename) as loadf:
        if loadf:
            model.load(loadf.rb())
    LOSS0 = 1e100
    tol = 0
    l = d = 0
    if not os.path.exists(vispath):
        os.mkdir(vispath)
    MPDrawInitializer(vispath)
    if isinstance(fdatagen, str):
        fdatagen = MPTwoAheadProducer(fdatagen)
    else:
        fdatagen = TwoAheadProducer(fdatagen)
    step = 0
    lrstep = 0
    if longrangecheck!=None and longrangeperiod==None:
        longrangeperiod = BATCHSTEP
    if longrangeperiod!=None:
        TRAINSETTINGS.LONGRANGEPERIOD = longrangeperiod
    while True:
        step += 1
        if step>totalsteps: break
        gen = fdatagen()
        loss,upd = [float(t) for t in ftrain(*gen)]
        l += loss
        d += upd
        sys.stdout.write('.')
        sys.stdout.flush()
        if step % TRAINSETTINGS.BATCHSTEP == TRAINSETTINGS.BATCHSTEP-1:
            print d,
            if TRAINSETTINGS.LEARNRATEVAR!=None:
                lval = TRAINSETTINGS.LEARNRATEVAR.get_value()
                if d>TRAINSETTINGS.LEARNRATETARGET*TRAINSETTINGS.BATCHSTEP: lval /= TRAINSETTINGS.LEARNADJUST
                else: lval *= TRAINSETTINGS.LEARNADJUST
                TRAINSETTINGS.LEARNRATEVAR.set_value(dtypeX(lval))
                print lval,
            if modrec!=None:
                modrec.R()
                modrec.Rlt(l)
                modrec.Rd()
            l = d = 0
            print "DRAW"
            #Draw model
            drawlayers = []
            layer = 0
            for i in model.paramlayers():
                if len(i.params)<1: continue
                layer += 1
                drawlayers.append((layer, i.params[0].get_value(), i.reshape if hasattr(i,'reshape') and i.reshape!=None else None))
            resplayers = fvis(*gen) if fvis!=None else []
            MPDrawWriter(drawlayers,resplayers)
            #Check validset
            if fcheckgen!=None and fcheck!=None:
                LOSS1 = 0.0
                for j in fcheckgen():
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    LOSS1 += fcheck(*j)
                print LOSS1
                if modrec!=None:
                    modrec.Rlv(LOSS1)
                if LOSS1>LOSS0:
                    print "Converge on validset"
                    tol+=1
                    if tol>TRAINSETTINGS.TOLTIMES:
                        sys.exit(0)
                else:
                    tol=0
                print "NEW LOSS",LOSS1
                LOSS0 = LOSS1
            if longrangecheck!=None:
                lrstep += 1
                if lrstep%TRAINSETTINGS.LONGRANGEPERIOD == TRAINSETTINGS.LONGRANGEPERIOD-1:
                    try:
                        result = longrangecheck()
                        modrec.Rfloat(result)
                    except KeyboardInterrupt: raise
                    except SystemExit: raise
                    except:
                        import traceback
                        traceback.print_exc()
            #Commit
            if modrec!=None:
                modrec.C()
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
    shapetmpl = list(slayers)
    #FIRST CONV
    slayers = [LCollect(ConvKeepLayer(rng, slayers[i], (channels[i], slayers[i].output_shape[1], RECEPTIVE,RECEPTIVE))) for i in range(SCALES)]
    #RECURRENT
    share = [None]*SCALES
    def buildrecur(inp,share):
        LPush()
        extinp = [None]+inp+[None]
        outs = []
        for i in range(SCALES):
            t=LCollect(*HIMBuilder(extinp[i],extinp[i+1],extinp[i+2],shapetmpl[i+1] if len(shapetmpl)<=i+1 else None))
            outs.append(LCollect(ConvKeepLayer(rng, t, (channels[i], t.output_shape[1], RECEPTIVE,RECEPTIVE), shareLayer = share[i], through=0)))
        if share[0]==None: share[:] = outs
        return LPop(), outs
    slayers = LCollect(*RepeatNetworkBuilder(slayers, buildrecur, REPEATS, share))
    #MERGE
    for i in range(SCALES-1,0,-1):
        t=LCollect(*HIMBuilder(None,slayers[i-1],slayers[i],shapetmpl[i]))
        slayers[i-1] = ConvKeepLayer(rng, t, (channels[i-1], t.output_shape[1], RECEPTIVE,RECEPTIVE))
    #OUTPUT
    outlayer = LCollect(ConvKeepLayer(rng, slayers[0], (shapetmpl[0].resp_shape[1], RECEPTIVE,RECEPTIVE), Nonlinear = False))
    return LPop(), outlayer




