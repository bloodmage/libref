import new
import numpy as np
import math

def RepeatNetworkBuilder(inp,build,repeat,*args,**kwargs):
    layers = []
    for it in range(repeat):
        thislayers, outs = build(inp,*args,**kwargs)
        layers.extend(thislayers)
        inp = outs
    return layers, outs

def HIMBuilder(low,mid,high,midtmpl):
    try:
        import fractallayer_nosquare as flayer
    except:
        import fractallayer as flayer
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
    dataqueue = Queue.Queue(10)
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
    module,func = invokestr.split(' ',1)
    if ' ' in func:
        func, extra = func.split(' ',1)
        extra = eval(extra)
        print "EXTRA:",extra
    else:
        extra = ()
    mod = __import__(module)
    fobj = getattr(mod,func)
    while True:
        queue.put(fobj(*extra))

cachedbalance = {}
def balancef(vmul,va):
    global cachedbalance
    if (vmul,va) not in cachedbalance:
        divmax = int(np.sqrt(vmul*va)/va)
        i=1
        for i in range(divmax,0,-1):
            if vmul%i==0: break
        cachedbalance[(vmul,va)] = i
    return cachedbalance[(vmul,va)]

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
    canvas[:]=0.5
    for i in range(flatblk.shape[0]):
        y = i // width
        x = i % width
        canvas[y*block.shape[2]+y:(y+1)*block.shape[2]+y,x*block.shape[3]+x:(x+1)*block.shape[3]+x] = flatblk[i]
    return np.array(canvas*255,np.uint8)

def DrawPatchLossdiff(block, blknorm = True):
    EPS = 1e-10
    flatblk = np.copy(block.reshape((-1,block.shape[2],block.shape[3])))
    if blknorm:
        flatblk = flatblk / (np.max(abs(flatblk)) + EPS)
    else:
        flatblk = flatblk / (np.max(abs(flatblk), axis=(1,2), keepdims=True) + EPS)

    width = math.ceil(math.sqrt(flatblk.shape[0]))
    height = (flatblk.shape[0] + width - 1) // width
    canvas = np.ones((height*block.shape[2]+height-1,width*block.shape[3]+width-1,3),'f')
    for i in range(flatblk.shape[0]):
        y = i // width
        x = i % width
        canvas[y*block.shape[2]+y:(y+1)*block.shape[2]+y,x*block.shape[3]+x:(x+1)*block.shape[3]+x,0] = np.maximum(flatblk[i],0)
        canvas[y*block.shape[2]+y:(y+1)*block.shape[2]+y,x*block.shape[3]+x:(x+1)*block.shape[3]+x,1] = np.maximum(-flatblk[i],0)
        canvas[y*block.shape[2]+y:(y+1)*block.shape[2]+y,x*block.shape[3]+x:(x+1)*block.shape[3]+x,2] = 0
    return np.array(canvas*255,np.uint8)

def DrawPatchRGB(block, blknorm = False):
    EPS = 1e-10
    assert block.shape[1]==3
    flatblk = np.copy(block.reshape((-1,block.shape[2],block.shape[3])))
    if blknorm:
        flatblk = (flatblk - np.min(flatblk)) / (np.max(flatblk) - np.min(flatblk)+EPS)
    else:
        flatblk = (flatblk-np.min(flatblk, axis=(1,2), keepdims=True)) / (np.max(flatblk, axis=(1,2), keepdims=True) - np.min(flatblk, axis=(1,2), keepdims=True)+EPS)

    width = math.ceil(math.sqrt(flatblk.shape[0]/3))
    height = (flatblk.shape[0]/3 + width - 1) // width
    canvas = np.zeros((height*block.shape[2]+height-1,width*block.shape[3]+width-1,3),'f')
    for i in range(flatblk.shape[0]/3):
        y = i // width
        x = i % width
        canvas[y*block.shape[2]+y:(y+1)*block.shape[2]+y,x*block.shape[3]+x:(x+1)*block.shape[3]+x] = flatblk[i*3:(i+1)*3].transpose(1,2,0)
    return np.array(canvas*255,np.uint8)

def __MPDrawFunc(vispath,queue):
    import PIL
    import PIL.Image
    import os
    while True:
        draws,resp = queue.get()
        try:
            for layer,param,reshape in draws:
                extra = None
                if reshape!=None:
                    if isinstance(reshape,list):
                        extra, reshape = reshape
                    param = param.reshape((-1,)+reshape[1:])
                if len(param.shape)==3:
                    param = param.reshape((1,)+param.shape)
                    #f = balancef(param.shape[0]*param.shape[1], param.shape[2])
                    #flatparam = param.reshape((param.shape[0]*param.shape[1]/f,param.shape[2]*f))
                    #PIL.Image.fromarray(np.array((flatparam-np.min(flatparam))/(np.max(flatparam)-np.min(flatparam))*255,np.uint8)).save(os.path.join(vispath,'layer_%s.png'%layer))
                    #continue
                if len(param.shape)==2:
                    PIL.Image.fromarray(np.array((param-np.min(param))/(np.max(param)-np.min(param))*255,np.uint8)).save(os.path.join(vispath,'layer_%s.png'%layer))
                    continue
                if extra == 'LOSS':
                    PIL.Image.fromarray(DrawPatchLossdiff(param)).save(os.path.join(vispath,'layer_%s.jpg'%layer), quality=100)
                else:
                    PIL.Image.fromarray(DrawPatch(param)).save(os.path.join(vispath,'layer_%s.jpg'%layer), quality=100)
            #Draw response
            layer = 0
            for i in resp:
                layer += 1
                if len(i.shape)==3:
                    i = i.reshape((1,)+i.shape)
                    #f = balancef(i.shape[0]*i.shape[1], i.shape[2])
                    #i = i.reshape((i.shape[0]*i.shape[1]/f,i.shape[2]*f))
                if len(i.shape)==2:
                    PIL.Image.fromarray(np.array((i-np.min(i))/(np.max(i)-np.min(i))*255,np.uint8)).save(os.path.join(vispath,'resp%s.png'%layer))
                    continue
                if i.shape[1]==3:
                    PIL.Image.fromarray(DrawPatchRGB(i)).save(os.path.join(vispath,'resp%s.jpg'%layer), quality=100)
                else:
                    if layer==len(resp):
                        PIL.Image.fromarray(DrawPatch(i, False)).save(os.path.join(vispath,'resp%s.jpg'%layer), quality=100)
                    else:
                        PIL.Image.fromarray(DrawPatch(i)).save(os.path.join(vispath,'resp%s.jpg'%layer), quality=100)
        except IOError:
            print "DRAW FAILED, IGNORE"

drawqueue = None
def MPDrawInitializer(vispath):
    global drawqueue
    if drawqueue!=None: return
    import multiprocessing
    dataqueue = multiprocessing.Queue(2)
    p = multiprocessing.Process(target=__MPDrawFunc, args=(vispath,dataqueue))
    p.daemon = True
    p.start()
    drawqueue = dataqueue
def MPDrawWriter(*data):
    global drawqueue
    drawqueue.put(data)

mpaheadprods = {}
def MPTwoAheadProducer(prodfuncstr):
    global mpaheadprods
    insts = 4
    if isinstance(prodfuncstr,(tuple,list)):
        prodfuncstr, insts = prodfuncstr
    if prodfuncstr not in mpaheadprods:
        import multiprocessing
        dataqueue = multiprocessing.Queue(2)
        for inst in range(insts):
            p = multiprocessing.Process(target=__MPWorkerFunc, args=(prodfuncstr, dataqueue))
            p.daemon = True
            p.start()
        def gen():
            return dataqueue.get()
        mpaheadprods[prodfuncstr] = TwoAheadProducer(gen)
    return mpaheadprods[prodfuncstr]

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
    from fractallayer import dtypeX
    if remotemonitor!=False:
        import modelrecord
        if remotemonitor==None: modrec = remotemonitor.Record()
        elif isinstance(remotemonitor, tuple): modrec = modelrecord.Record(*remotemonitor)
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
    if vispath!=None:
        if not os.path.exists(vispath):
            os.mkdir(vispath)
        MPDrawInitializer(vispath)
    if isinstance(fdatagen, (str,tuple,list)):
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
        if TRAINSETTINGS.TOTALSTEPS!=None and step>TRAINSETTINGS.TOTALSTEPS: break
        while True:
            try:
                gen = fdatagen()
                loss,upd = [float(t) for t in ftrain(*gen)]
                break
            except KeyboardInterrupt:raise
            except SystemExit:raise
            except:
                import traceback
                traceback.print_exc()
                sys.stdout.write('*')
                continue
        l += loss
        d += upd
        sys.stdout.write('.')
        sys.stdout.flush()
        if step % TRAINSETTINGS.BATCHSTEP == TRAINSETTINGS.BATCHSTEP-1:
            print d,l,
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
            if vispath!=None:
                print "DRAW"
                #Draw model
                drawlayers = []
                layer = 0
                for i in model.paramlayers():
                    if len(i.params)<1: continue
                    if len(i.params)>2:
                        if hasattr(i,'reshape'):
                            reshape = i.reshape
                        else:
                            reshape = [None]*len(i.params)
                        for j,rj in zip(i.params,reshape):
                            s = j.get_value()
                            vsh = [i for i in s.shape if i>1]
                            if len(vsh)<2: continue
                            layer += 1
                            drawlayers.append((layer, s, rj))
                    else:
                        layer += 1
                        drawlayers.append((layer, i.params[0].get_value(), i.reshape if hasattr(i,'reshape') and i.reshape!=None else None))
                resplayers = fvis(*gen) if fvis!=None else []
                MPDrawWriter(drawlayers,resplayers)
            else:
                print
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

