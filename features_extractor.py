#Signal Features extractor project
import numpy as np

class Chromosome():

    def __init__(self,signal,freq):

        self = self
        self.signal = Signal
        self.freq = freq

    def fit(array):
        pass

    def lat(self):
        """ Returns latency of signal """
        return np.argmax(np.abs(self.signal))

    def amp(self):
        """ Returns amplitude of signal """
        return self.signal[np.argmax(np.abs(sself.signal))]

    def lar(self):
        """ Returns latency to amplitude ratio """
        return self.lat(self.signal)/self.amp(self.signal)

    def aamp(self):
        """ Returns absolute amplitude of signal """
        return np.max(np.abs(self.signal))

    def alar(self):
        """ Returns an absolute latency to amplitude ratio  """
        return np.abs(self.lar(self.signal))

    def par(self):
        """ Returns positive area, sum of positive values of signal """
        return np.sum([n for n in self.signal if n > 0])

    def nar(self):
        """ Returns negative area, sum of negative values of signal """
        return np.sum([n for n in self.signal if n < 0])

    def tar(self):
        """ Returns total area, sum of all values of signal """
        return np.sum(self.signal)

    def atar(self):
        """ Returns absolute total area """
        return np.abs(self.tar())

    def taar(signal):
        """ Returns total absolute area """
        return np.sum(np.abs(self.signal))

    def aass(self):
        """ Returns average absolute signal slope """
        n_probes = len(self.signal)
        r = 1/self.freq
        return np.sum([np.abs((self.signal[n+1])-self.signal[n])/r for
                      n in range(n_probes-1)])/n_probes

    def pp(self):
        """ Returns peak-to-peak """
        return np.max(self.signal)-np.min(self.signal)

    def ppt(self):
        """ Returns peak-to-peak time window """
        return np.argmax(self.signal)-argmin(self.signal)

    def pps(self):
        """ Returns peak-to-peak slope """
        return self.pp()/self.ppt()

    def zc(self):
        """ Returns zero crossings """
        return np.count_nonzero(np.where(np.diff(np.sign(self.signal)))[0])
        #Solution taken from stack overflow
        # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-ch
        # anges-in-python

    def zcd(self):
        """ Returns zero crossings density """
        return np.count_nonzero(np.where(
                                np.diff(np.sign(self.signal)))[0])/self.pp()
        #Solution taken from stack overflow

    #TODO: Slope sign alterationsdef SSA(signal):
    def SSA(self):
        """ Returns slope signs alterations """
        signal = self.signal.copy()
        return np.sum(0.5 *(np.abs(
        [((signal[n-1]-signal[n])/np.abs(
        signal[n-1]-signal[n]) + (signal[n+1]-signal[n])/np.abs(
        signal[n+1]-signal[n])) for n in range(len(signal)-1)] )))
