import numpy.fft as fft
import numpy as np
import os

whiteconfs = {}

def white(arr4d, cachename):
    if os.path.exists(cachename+'.npz'):
        cache = np.load(cachename+'.npz')
        subterm = cache['sub']
        divterm = cache['div']
        fftterm = cache['fft']
        whiteconfs[cachename] = [subterm,divterm,fftterm]
        arr4d = (arr4d - subterm) * divterm
        farr4d = fft.fft2(arr4d)
        farr4d = farr4d * fftterm
        arr4d = fft.ifft2(farr4d)
        return arr4d

    subterm = np.mean(arr4d,0,keepdims = True)
    divterm = 1.0/(arr4d.std(axis=(2,3),keepdims=True)+1e-10)
    arr4d = (arr4d - subterm) * divterm
    farr4d = fft.fft2(arr4d)
    fftterm = 1.0/(np.sqrt(np.mean(np.abs(farr4d)**2,axis=0))+1e-10)
    farr4d = farr4d * fftterm
    arr4d = np.real(fft.ifft2(farr4d))
    np.savez(cachename+'.npz',sub=subterm,div=divterm,fft=fftterm)
    whiteconfs[cachename] = [subterm,divterm,fftterm]
    return arr4d

def unwhite(arr4d, cachename):
    assert cachename in whiteconfs
    subterm, divterm, fftterm = whiteconfs[cachename]
    farr4d = fft.fft2(arr4d)
    farr4d = farr4d / fftterm
    arr4d = np.real(fft.ifft2(farr4d))
    arr4d = arr4d / divterm + subterm
    return arr4d

if __name__=="__main__":
    import numpy.random as npr
    d = npr.ranf((10,1,20,20))
    dw = white(d,'test')
    d2 = unwhite(dw,'test')
    os.unlink('test.npz')
    print d
    print dw
    print np.sum(np.abs(d2-d))

