#coding=gbk
import layerbase
import Queue
import threading
import requests
import time
starttime = time.time()
import numpy as np
import math
import cStringIO as sio
from itertools import izip
import os
import sys
try:
    import resource
except:
    resource = None
import platform
import uuid
import commands
import socket
import atexit
import json
import base64
import itsdangerous
import PIL
from layerbase import DrawPatch

DATAREC = [
        ('sum', 'float'),
        ('abssum', 'float'),
        ('sqrsum', 'float'),
        ('max', 'float'),
        ('min', 'float'),
        ('absmax', 'float'),
        ('updatedeltaabs', 'float'),
        ('updatedeltasqr', 'float'),
        ('updatedeltasum', 'float'),
        ('filter', 'png'),
    ]

VISRESPREC = [
        ('resptrain', 'jpg'),
        ('respvalid', 'jpg'),
    ]

LOSSREC = [
        ('losstrain', 'float'),
        ('lossvalid', 'float'),
    ]

N = 61234458376886086861524070385274672740778091784697328983823014963978384987221695735286049226673298932707
G = 5

once = set()
def onlyonce(val):
    global once
    if val not in once:
        once.add(val)
        return True
    return False

def _gethash():
    import sys
    import binascii
    fnames = [i.__file__ for i in sys.modules.values() if hasattr(i,'__file__')]
    hashs = [binascii.crc32(file(i,'rb').read()) for i in fnames]
    return reduce(lambda x,y:x^y,hashs)

def _bestsplit(val):
    t = int(math.sqrt(val))
    for i in range(t,0,-1):
        if val % i == 0:
            return i,val/i
    return 1,val

try:
    from blessings import Terminal


    class PatchedStdout:
        def __init__(self, mesg=''):
            self.org = sys.stdout
            self.term = Terminal()
            self.newline = True
            self.mesg = mesg
            self.alarm('Header Patch Enabled')
            sys.stdout=self
        
        def alarm(self, data):
            self.write(self.term.bold+self.term.yellow+data+self.term.normal)
            self.write('\n')
        
        def write(self, data):
            dlines = data.split('\n')
            for i in dlines[:-1]:
                if self.newline:
                    self.org.write(self.term.red+self.term.bold)
                    self.org.write('[%s] '%self.mesg)
                    self.org.write(self.term.normal)
                self.org.write(i)
                self.org.write('\n')
                self.newline = True
            if len(dlines[-1])!=0:
                if self.newline:
                    self.org.write(self.term.red+self.term.bold)
                    self.org.write('[%s] '%self.mesg)
                    self.org.write(self.term.normal)
                    self.org.flush()
                    self.newline = False
                self.org.write(dlines[-1])
        
        def flush(self):
            self.org.flush()
        
        def setmesg(self, mesg):
            self.mesg = mesg
        
except:
    class PatchedStdout:
        def __init__(self, mesg=''):
            self.org = sys.stdout
            self.newline = True
            self.mesg = mesg
            self.alarm('Header Patch Enabled without Blessings')
            sys.stdout = self
        
        def alarm(self, data):
            self.org.write(data)
            self.org.write('\n')
        
        def write(self, data):
            dlines = data.split('\n')
            for i in dlines[:-1]:
                if self.newline:
                    self.org.write('[%s] '%self.mesg)
                self.org.write(i)
                self.org.write('\n')
                self.newline = True
            if len(dlines[-1])!=0:
                if self.newline:
                    self.org.write('[%s] '%self.mesg)
                    self.org.flush()
                    self.newline = False
                self.org.write(dlines[-1])
        
        def flush(self):
            self.org.flush()
        
        def setmesg(self, mesg):
            self.mesg = mesg

def _mystatrec():
    "记录程序统计信息"
    #main位置，GPU，内存，起止时间，机器名，IP，磁盘，load
    import __main__
    stat = {}
    #PATH
    if hasattr(__main__,'__file__'):
        stat['filepath'] = os.path.abspath(__main__.__file__)
    #GPU
    if 'theano' in sys.modules:
        stat['device'] = sys.modules['theano'].config.device
    #MEM
    if resource!=None:
        stat['mem'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            stat['status'] = file('/proc/self/status').read()
        except:
            pass
    #TIME
    stat['walltime'] = time.time() - starttime
    try:
        p = file('/proc/self/stat').read().strip().split()
        stat['usertime'] = int(p[14])
        stat['systime'] = int(p[15])
    except:
        pass
    #MACHINE
    stat['machine'] = platform.node()
    #IP/MAC
    stat['mac'] = uuid.getnode()
    try:
        stat['ips'] = commands.getoutput('/sbin/ifconfig | grep -i "inet" | grep -iv "inet6" | awk {\'print $2\'} | sed -ne \'s/addr\:/ /p\'').replace('\n ',',')
        stat['detailip'] = commands.getoutput('ip address show')
    except:
        stat['ips'] = socket.gethostbyname(socket.gethostname())
    #DISKFREE
    #LOAD
    try:
        stat['diskfree'] = commands.getoutput('df')
        stat['load'] = os.getloadavg()
    except:
        pass
    return stat

class AsyncRest:
    "异步REST派遣"
    def __init__(self, calls = 10000, parallel = 5):
        "启动多个REST服务程序"
        self.maxcall = calls
        self.parallel = parallel
        self.pool = Queue.Queue(self.maxcall)
        #Init thread pool
        def work():
            while True:
                url, data, callback = self.pool.get()
                while True:
                    try:
                        r = requests.post(url, data = data)
                        if r.status_code != 200: continue
                    except KeyboardInterrupt:
                        raise
                    except requests.exceptions.RequestException:
                        continue
                    break
                if callback!=None:
                    try:
                        callback(r.text)
                    except KeyboardInterrupt:
                        raise
                    except:
                        import traceback
                        traceback.print_exc()
                self.pool.task_done()
        pool = [threading.Thread(target=work) for i in range(parallel)]
        for i in pool:
            i.daemon = True
            i.start()

    def restcall(self, url, data, callback = None):
        "提交一个REST请求"
        self.pool.put((url, data, callback))
    
    def blockcall(self, url, data):
        val = []
        ev = threading.Event()
        def setev(txt):
            val.append(txt)
            ev.set()
        self.pool.put((url, data, setev))
        
        while not ev.isSet():
            ev.wait(1)
        return val[0]
    
    def join(self):
        self.pool.join()

class Record:
    "实验记录，将进行的实验数据传至服务器"

    def __init__(self, server = 'exp.zysdomain.tk'):
        self.dirty = {}
        self.dataupload = None
        self.seq = 0
        self.server = server
        self.rest = AsyncRest()
        atexit.register(self.S)

    def _makerecmap_data(self, datamapid):
        "内部——生成数据记录列"
        recmap = self.meta['recmap']
        ret = []
        for name,datatype in DATAREC:
            ret.append(len(recmap))
            recmap.append({
                'dataid':datamapid,
                'func':name,
                'datatype':datatype
                })
        return ret

    def _makerecmap_loss(self, datamapid):
        "内部——生成LOSS记录列"
        recmap = self.meta['recmap']
        ret = []
        for name,datatype in LOSSREC:
            ret.append(len(recmap))
            recmap.append({
                'dataid':datamapid,
                'func':name,
                'datatype':datatype
                })
        return ret
    
    def _makerecmap_visresp(self, datamapid):
        "内部——生成输出记录列"
        recmap = self.meta['recmap']
        ret = []
        for name,datatype in VISRESPREC:
            ret.append(len(recmap))
            recmap.append({
                'dataid':datamapid,
                'func':name,
                'datatype':datatype
                })
        return ret
    
    def genmeta(self, model, sameranks = []):
        "根据网络结构生成元信息"
        self.model = model
        self.layers = model.layers
        #Extract each layer's name
        try:
            raise ZeroDivisionError
        except ZeroDivisionError:
            f = sys.exc_info()[2].tb_frame.f_back
        namelist = f.f_locals

        for i in self.layers:
            for j in namelist.keys():
                if namelist[j] is i:
                    #Name each layer
                    i.rec__name = j

        #Create metadata
        struct = []
        datamap = []
        recmap = []
        self.meta = {'struct':struct,'datamap':datamap,'recmap':recmap}
        structback = []
        for i in self.layers:
            if isinstance(i,layerbase.Layer):
                ast = {
                   'type':i.__class__.__name__,
                   'name':i.rec__name,
                   'outputs': layerbase.Layer.linkstruct[i],
                   'outshape': i.output_shape,
                   'hasparam': 0,
                   'hasloss': 0,
                   'hasvis': 0}
                if isinstance(i,layerbase.Param):
                    i.rec__dataparam_map = []
                    i.rec__dataparam = []
                    for j in i.params:
                        i.rec__dataparam_map.append(len(datamap))
                        i.rec__dataparam.append(self._makerecmap_data(len(datamap)))
                        params = {'dataname': j.name,
                                'layerid': len(struct),
                                'filtershape': j.get_value().shape}
                        datamap.append(params)
                    ast['hasparam'] = 1
                if isinstance(i,layerbase.VisLayer):
                    i.rec__visparam_map = len(datamap)
                    i.rec__visparam = self._makerecmap_visresp(len(datamap))
                    params = {'dataname': '_vis_'+i.rec__name,
                            'layerid': len(struct),
                            'filtershape': i.output_shape}
                    datamap.append(params)
                    ast['hasvis'] = 1
                if isinstance(i,layerbase.LossLayer):
                    self.loss = self._makerecmap_loss(len(datamap))
                    params = {'dataname': '_loss_'+i.rec__name,
                            'layerid': len(struct),
                            'filtershape': (1,)}
                    datamap.append(params)
                    ast['hasloss'] = 1
                i.rec__struct_id = len(struct)
                struct.append(ast)
                structback.append(i)
        
        refs = [0] * len(struct)
        
        for i in struct:
            i['outputs'] = [j.rec__struct_id for j in i['outputs'] if hasattr(j,'rec__struct_id')]
            for j in i['outputs']:
                refs[j] += 1
        
        #整理形状信息
        ref0 = [i for i in range(len(struct)) if refs[i]==0]
        masked = set()
        nrank = 0
        while ref0!=[]:
            while ref0!=[]:
                for i in ref0:
                    masked.add(i)
                    struct[i]['rank'] = nrank
                    for j in struct[i]['outputs']:
                        refs[j] -= 1
                ref0 = [i for i in range(len(struct)) if refs[i]==0 and i not in masked and (isinstance(structback[i],layerbase.VisSamerank) or structback[i] in sameranks)]
            ref0 = [i for i in range(len(struct)) if refs[i]==0 and i not in masked]
            nrank += 1
        
        #提交元信息，创建实验
        self.newexp()
    
    def newexp(self):
        "新实验"
        a = 1
        for i in os.urandom(40):
            a = a*256 + ord(i)
        A = pow(G,a,N)
        from select import select
        print "Experiment name:",
        try:
            rlist, _, _ = select([sys.stdin], [], [], 30)
            if rlist:
                naming = sys.stdin.readline().strip()
            else:
                print "Use default name"
                naming = ''
        except KeyboardInterrupt:
            raise
        except:
            naming = raw_input()
        #Call service
        val = self.rest.blockcall('http://'+self.server+'/newexperiment',{'meta':json.dumps(self.meta),'serieshash':_gethash(), 'A':str(A), 'naming': naming})
        vald = json.loads(val)
        self.expid = vald['id']
        self.expname = vald['name']
        B=int(vald['B'])
        self.exp_key = str(pow(B,a,N))
        self.signer = itsdangerous.Signer(self.exp_key)
        _stat = _mystatrec()
        PatchedStdout(self.expname.encode('utf-8')+(u'(%s,%s,%s)'%(self.expid,_stat['device'],os.path.split(_stat['filepath'])[1])).encode('utf-8'))
        
    def R(self):
        "记录数值"
        self._newrec()
        for i in self.layers:
            if isinstance(i,layerbase.Param):
                for idx in range(len(i.params)):
                    param = i.params[idx]
                    val = param.get_value()
                    d1 = val.sum()
                    d2 = np.abs(val).sum()
                    d3 = (val*val).sum()
                    d4 = np.max(val)
                    d5 = np.min(val)
                    d6 = np.max(np.abs(val))
                    d1,d2,d3,d4,d5,d6 = map(float, (d1,d2,d3,d4,d5,d6))
                    dp = i.rec__dataparam[idx]
                    self.datastore.extend([
                        [dp[0], d1],
                        [dp[1], d2],
                        [dp[2], d3],
                        [dp[3], d4],
                        [dp[4], d5],
                        [dp[5], d6]])

    def Rlt(self, loss):
        "记录训练集loss"
        self._newrec()
        self.datastore.append([self.loss[0], float(loss)])

    def Rlv(self, loss):
        "记录验证集loss"
        self._newrec()
        self.datastore.append([self.loss[1], float(loss)])

    def Rd(self):
        "记录层信息有效更新值"
        self._newrec()
        for i in self.layers:
            if isinstance(i,layerbase.Param):
                if not hasattr(i,'rec__lastval'):
                    #Store thisval as lastval
                    i.rec__lastval = [np.copy(j.get_value()) for j in i.params]
                else:
                    thisval = [np.copy(j.get_value()) for j in i.params]
                    updates = [a-b for a,b in zip(thisval,i.rec__lastval)]
                    for idx in range(len(i.params)):
                        dp = i.rec__dataparam[idx]
                        #Record update
                        self.datastore.extend([
                            [dp[6], float(np.sum(np.abs(updates[idx])))],
                            [dp[7], float(np.sum(updates[idx]*updates[idx]))],
                            [dp[8], float(np.sum(updates[idx]))]])
                        #Store thisval as lastval
                        i.rec__lastval = thisval
    
    def Rv(self):
        "可视化记录层"
        self._newrec()
        #Render images for each W and b
        for i in self.layers:
            if isinstance(i,layerbase.Param):
                for idx in range(len(i.params)):
                    param = i.params[idx]
                    val = param.get_value()
                    img = None
                    #4 condition:
                    if len(val.shape)==4:
                        #Conv (4x)
                        img = PIL.Image.fromarray(DrawPatch(val))
                    elif len(val.shape)==5:
                        #OutputShape, InputShape, SizeH, SizeW, DimMixture
                        #Unshare (5x)
                        nshape = (val.shape[0],)+_bestsplit(val.shape[1])+(val.shape[2],val.shape[3],i.filter_shape[2],i.filter_shape[3])
                        img = PIL.Image.fromarray(DrawPatch(val.reshape(nshape).transpose((0,3,4,1,5,2,6)).reshape((nshape[0],nshape[3]*nshape[4],nshape[1]*nshape[5],nshape[2]*nshape[6])), blknorm = False))

                    elif len(val.shape)==2:
                        if hasattr(i,'reshape') and i.reshape!=None:
                            #ReshapeMatrix (2x, with_reshape)
                            img = PIL.Image.fromarray(DrawPatch(val.reshape((-1,)+i.reshape[1:])))
                        else:
                            #Matrix (2x)
                            img = IL.Image.fromarray(np.array((val-np.min(val))/(np.max(val)-np.min(val))*255,np.uint8))
                    else:
                        if onlyonce(i):
                            print "Warning: unhandled shape",val.shape
                    if img!=None:
                        #Record image
                        buf = sio.StringIO()
                        img.save(buf,'PNG')
                        self.imgstore.append([
                                i.rec__dataparam[idx][9],'.png', buf.getvalue()
                            ])

    
    def _enumvis(self):
        "内部——枚举输出"
        for i in self.layers:
            if isinstance(i,layerbase.Param):
                yield i

    def Rrt(self,resps):
        "可视化记录输入输出(训练集)"
        self._newrec()
        for l,i in izip(_enumvis(),resps):
            #Draw to img
            if len(i.shape)!=4:
                img = PIL.Image.fromarray(np.array((i-np.min(i))/(np.max(i)-np.min(i))*255,np.uint8))
            else:
                img = PIL.Image.fromarray(DrawPatch(i[0:1]))
            bug = sio.StringIO()
            img.save(buf,'JPEG',quality=100)
            self.imgstore.append([
                l.rec__visparam[0],'.jpeg', buf.getvalue()
                ])

    def Rrv(self,resps):
        "可视化记录输入输出(验证集)"
        self._newrec()
        for l,i in izip(_enumvis(),resps):
            #Draw to img
            if len(i.shape)!=4:
                img = PIL.Image.fromarray(np.array((i-np.min(i))/(np.max(i)-np.min(i))*255,np.uint8))
            else:
                img = PIL.Image.fromarray(DrawPatch(i[0:1]))
            bug = sio.StringIO()
            img.save(buf,'JPEG',quality=100)
            self.imgstore.append([
                l.rec__visparam[1],'.jpeg', buf.getvalue()
                ])

    def _newrec(self):
        "内部——新增记录结构"
        if self.dataupload == None:
            self.seq += 1
            self.datastore = []
            self.imgstore = []
            self.dataupload = {
                    'serialnum': self.seq,
                    'date': time.time(),
                    'data': self.datastore
                    }

    def C(self):
        "进行一个批次提交，附加提交瞬间内存CPU等信息"
        statupload = _mystatrec()
        dataupload = self.dataupload
        #Call api to load data/stat/single image
        self.rest.restcall('http://'+self.server+'/updateexp', {'expid':self.expid, 'info':self.signer.sign(json.dumps(statupload))})
        self.rest.restcall('http://'+self.server+'/record', {'expid':self.expid, 'data':self.signer.sign(json.dumps(dataupload))})
        for meta,type_,data in self.imgstore:
            self.rest.restcall('http://'+self.server+'/recordimg', {'expid':self.expid,
                                                                    'data':self.signer.sign(
                                                                        json.dumps({'meta':meta, 'serial':dataupload['serialnum'], 'date':dataupload['date'], 'name':type_, 'b64data':base64.b64encode(data)})
                                                                    )}
                               )
        self.dataupload = None

    def S(self):
        "等待完成所有挂起的提交"
        import sys
        if not isinstance(sys.exc_info()[1],KeyboardInterrupt):
            time.sleep(20)

if __name__=="__main__":
    PatchedStdout('1234')
    print "test"
    for j in range(10):
        for i in range(j):
            print ",",
            sys.stdout.flush()
            time.sleep(1)
        print
        time.sleep(1)
    

