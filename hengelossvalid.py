import theano
import theano.tensor as T

def binaryloss2(layerout, label, thresholdlayer, thresholdlabel):

    obinary = T.switch(T.gt(layerout, thresholdlayer), 1.0, 0.0)
    lbinary = T.switch(T.gt(label, thresholdlabel), 1.0, 0.0)

    positives = T.sum(lbinary, axis=(0, ))
    negatives = T.sum(1-lbinary, axis=(0, ))

    tpos = T.sum(lbinary * obinary, axis=(0, ))
    fneg = T.sum(lbinary * (1-obinary), axis=(0, ))
    tneg = T.sum((1-lbinary) * (1-obinary), axis=(0, ))
    fpos = T.sum((1-lbinary) * obinary, axis=(0, ))

    precision = tpos / (tpos + fpos)
    recall = tpos / (tpos + fneg)
    F = 2*precision*recall / (precision + recall)

    return obinary, tpos,fpos,tneg,fneg, F
def binaryloss(layerout, label, thresholdlayer, thresholdlabel):

    obinary = T.switch(T.gt(layerout, thresholdlayer), 1.0, 0.0)
    lbinary = T.switch(T.gt(label, thresholdlabel), 1.0, 0.0)

    positives = T.sum(lbinary, axis=(0, 2, 3))
    negatives = T.sum(1-lbinary, axis=(0, 2, 3))

    tpos = T.sum(lbinary * obinary, axis=(0, 2, 3))
    fneg = T.sum(lbinary * (1-obinary), axis=(0, 2, 3))
    tneg = T.sum((1-lbinary) * (1-obinary), axis=(0, 2, 3))
    fpos = T.sum((1-lbinary) * obinary, axis=(0, 2, 3))

    precision = tpos / (tpos + fpos)
    recall = tpos / (tpos + fneg)
    F = 2*precision*recall / (precision + recall)

    return obinary, tpos,fpos,tneg,fneg, F
