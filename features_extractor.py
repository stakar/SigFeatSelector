#Signal Features extractor project
import numpy as np

def lat(signal):
    return np.argmax(np.abs(signal))

def amp(signal):
    return signal[np.argmax(np.abs(signal))]

def lar(signal):
    return lat(signal)/amp(signal)

def aamp(signal):
    return np.max(np.abs(signal))

def alar(signal):
    return np.abs(lar(signal))

def par(signal):
    return np.sum([n for n in signal if n > 0])

def nar(signal):
    return np.sum([n for n in signal if n < 0])

def tar(signal):
    return np.sum(signal)

#TODO: Average absolute signal slope
# def aass(signal):

def pp(signal):
    return np.max(signal)-np.min(signal)

def ppt(signal):
    return np.argmax(signal)-argmin(signal)

def zc(signal):
    return np.count_nonzero(np.where(np.diff(np.sign(signal)))[0])
    #Solution taken from stack overflow

def zcd(signal):
    return np.count_nonzero(np.where(np.diff(np.sign(signal)))[0])/pp(signal)
    #Solution taken from stack overflow

#TODO: Slope sign alterations
# def SSA(signal):
