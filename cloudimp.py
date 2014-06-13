#coding=gbk
"""
云端Import & 自动更新Python脚本
"""

import sys
import os
import urllib2
import md5
import urlparse

def cloudimport(url):
    fname = urlparse.urlsplit(url).path.split('/')[-1]
    hexpart = md5.md5(url).hexdigest()[:8]
    totalname = fname.split('.')[-1]+'_'+hexpart

    doc = urllib2.urlopen(url).read()
    
    if (not os.path.exists(totalname+'.py')) or doc!=file(totalname+'.py','rb').read():
        print "REFRESH MODULE",totalname
        file(totalname+'.py','wb').write(doc)
    
    return __import__(totalname)

