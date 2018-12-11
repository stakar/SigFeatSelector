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

time = 3
freq = 256
n_probes = time*freq
r = time/n_probes

def aass(signal,r,n_probes):
    return np.sum([np.abs((signal[n+1])-signal[n])/r for
                  n in range(n_probes-1)])/n_probes

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
# SSA:
# np.sum( 0.5 * ( np.abs( ([signal[n-1]-signal[n])/np.abs(signal[n-1]-signal[n]) + for ] )))
