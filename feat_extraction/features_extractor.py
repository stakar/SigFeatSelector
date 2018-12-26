#Signal Features extractor project
import numpy as np

class Chromosome():

    def __init__(self,genotype=[1 for n in range(17)],freq = 256):


        """
        This class creates chromosome, by extracting the features of signal and
        returning those as a set.

        ----------
        Attributes
        ----------

        genotype : array[n_genes]
        Set of genes, 0 and 1, each mapping one of features

        freq : int
        Frequency of signal that is going to be fitted to model

        ----------
        References
        ----------

        Abotoolebi et al. 2008
        A new approach for EEG feature extraction in P300-based lie
        detection

        """

        self = self
        self.genotype = genotype
        self.freq = freq

    def fit(self,signal):
        """ Fits the model to signal """
        self.signal = signal

        genes = [self.lat,self.amp,self.lar,self.aamp,self.alar,
        self.par,self.nar,self.tar,self.atar,self.taar,self.aass,
        self.pp,self.ppt,self.pps,self.zc,self.zcd,self.ssa]

        self.chromosome = [genes[n]() for n in range(17) if self.genotype[n] > 0]

    def transform(self):
        """ Returns chromosome """
        return self.chromosome

    def fit_transform(self,signal):
        """ Fits the model to data and returns chromosome """
        self.fit(signal)
        return self.transform()

    def lat(self):
        """ Returns latency of signal """
        return np.argmax(np.abs(self.signal))

    def amp(self):
        """ Returns amplitude of signal """
        return self.signal[np.argmax(np.abs(self.signal))]

    def lar(self):
        """ Returns latency to amplitude ratio """
        return self.lat()/self.amp()

    def aamp(self):
        """ Returns absolute amplitude of signal """
        return np.max(np.abs(self.signal))

    def alar(self):
        """ Returns an absolute latency to amplitude ratio  """
        return np.abs(self.lar())

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

    def taar(self):
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
        return np.argmax(self.signal)-np.argmin(self.signal)

    def pps(self):
        """ Returns peak-to-peak slope """
        return self.pp()/self.ppt()

    def zc(self):
        """ Returns zero crossings
        Solution partly taken from stack overflow:
        https://stackoverflow.com/questions/3843017/efficiently-detect-sign-chan
        ges-in-python """
        return np.count_nonzero(np.where(np.diff(np.sign(self.signal)))[0])
        #Solution taken from stack overflow
        # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-ch
        # anges-in-python

    def zcd(self):
        """ Returns zero crossings density
        Solution partly taken from stack overflow:
        https://stackoverflow.com/questions/3843017/efficiently-detect-sign-chan
        ges-in-python
         """
        return np.count_nonzero(np.where(
                                np.diff(np.sign(self.signal)))[0])/self.pp()
        #


    def ssa(self):
        """ Returns slope signs alterations.
        There is little problem, for the function returns nan when passed 0 valu
        e amongst others in signal. However, in my system it should not have bee
        n problem, as BakSys returns signal containing no zero values due to ene
        rgy variance extraction and latter savitzky-golay smoothing """
        signal = self.signal.copy()
        return np.sum(0.5 *(np.abs(
        [((signal[n-1]-signal[n])/np.abs(
        signal[n-1]-signal[n]) + (signal[n+1]-signal[n])/np.abs(
        signal[n+1]-signal[n])) for n in range(len(signal)-1)] )))

if __name__ == '__main__':
    time = 1
    freq = 256
    t = np.linspace(0,time,time*freq)
    signal = np.sin(t*np.pi*52)
    chrom = [1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1]
    cr = Chromosome(chrom)
    # cr.fit(signal)
    print(cr.fit_transform(signal))
