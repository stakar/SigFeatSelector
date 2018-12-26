# This file is containing a class, that perform all modules for
# Hovagim Bakardjian system that serves for feature extraction and command
# classification in SSVEP based BCI.
# Also, it has an built-in FFT features extractor

# Version 3.4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.stats import mode
from .amuse import AMUSE


def load_data_path(path,sep=' '):
    """
    Load data from input path and extract artifacts.

    Parameters
    ----------
    path : str
        an input path that leads to the EEG data on which operations should be
        performed. Data should be with csv or tsv extension,
        with shape [n_channels,n_probes]
    """
    data = np.loadtxt(path,delimiter=sep)
    return data


class BakardjianSystem(object):

    def __init__(self, extract=False, freq=256,
                 sep = ' ', channels = [15,23,28],threeclass = True,
                 seconds = 1):
        """
        Bakardjian System

        Bakarjian System is a class that takes EEG data and performs signal
        analysis proper to Bakardjian system.

        Parameters
        ----------

        sep : str
            a separator used in data that are supposed to be load

        extract : bool
            decision whether to extract comoponents or not

        freq : int
            sampling frequency of data

        sep : int
            delimiter used in data

        channels : list
            channels on which analysis is supposed to be performed

        threeclass : boolean
            decision whether perform two- or three class classification.

        seconds : int
            length of time window in seconds

        Attributes
        ----------

        data : numpy array

        featFFT : numpy array

        morphfeat : numpy array

        decision : int

        References
        ----------

        Hovagim Bakardjian, Optimization of steady-state
        visual responses for robust brain-computer interfaces. 2010

        """

        self.self = self
        self.extract = extract
        self.freq = freq
        self.channels = channels
        self.sep = sep
        self.threeclass = threeclass
        self.seconds = seconds

    @staticmethod
    def _extract_components(data):
        amuse = AMUSE(data,data.shape[0],1)
        return amuse.sources

        if self.extract == True:
            data = self._extract_components(data)

        self.data = data[self.channels,:]

    def load_data(self,data):
        """
        Load data from input and extract artifacts.

        """
        if self.extract == True:
            data = self._extract_components(data)

        self.data = data[self.channels,:]

    @staticmethod
    def __filtering(data,low,high,freq):

        """
        Filter the data using band-pass filter.

            Parameters
            ----------

            data : array
                Array of data, that is signal supposed to be filtered.

            low  : float
                Lower band of frequency

            high : float
                Higher band of frequency

            freq : int
                Frequency of sampling

        """
        bplowcut = low/(freq*0.5)
        bphighcut = high/(freq*0.5)
        [b,a] = sig.butter(N=3,Wn=[bplowcut,bphighcut],btype='bandpass')
        filtered = sig.filtfilt(b,a,data)

        return filtered

    def __matfilt(self,data,low,high,freq):

        """
        Filter the matrix of data using built-in band-pass filter.

            Parameters
            ----------

            data : matrix array-like, shape [n_channels,n_probes]
                Matrix of data

            low  : float
                Lower band of frequency

            high : float
                Higher band of frequency

            freq : int
                Frequency of sampling
        """
        C, P = data.shape
        result = np.zeros([C,P])

        for n in range(C):
            result[n,:] = self.__filtering(data[n,:],low,high,freq)

        return result

    def bank_of_filters(self):

        """
        Filter each channel using narrow bandpass filters.
        """

        x = self.data
        X = self.__matfilt(x,7.9,8.1,self.freq)
        Y = self.__matfilt(x,13.9,14.1,self.freq)
        self.data = np.array([X,Y])

        if self.threeclass == True:
            Z = self.__matfilt(x,27.9,28.1,self.freq)
            self.data = np.array([X,Y,Z])

    def variance_analyzer(self):

        """
        Extract energy variance of signal.
        """

        self.data = abs(self.data)

    def smoothing(self):

        """
        Smooth the data
        """

        F,C,P = self.data.shape
        X = np.zeros((F,C,P))

        for n in range(0,F):
            for i in range(C):
                x = self.data[n,[i]]
                X[n,[i]] = sig.savgol_filter(x,polyorder=2,
                                            window_length =(self.freq*self.seconds)-1,
                                            deriv=0,mode='wrap')

        self.data = X

    def integrating(self):

        """
        Integrate channels for each analyzed frequency.
        """

        data = self.data
        F,C,P = data.shape
        result = np.zeros((F,1,P))

        for n in range(F):
            for z in range(P):
                result[n,0,[z]] = np.mean(data[n,:,[z]])

        self.data = result


    def normalize(self):

        """
        Normalize the data.
        """

        F,C,P = self.data.shape
        S = np.zeros((1,C,P))

        for n in range(F):
            S += self.data[n]

        for n in range(F):
            self.data[n] = self.data[n]/S

    def fit(self,data):
        """
        This method performs all modules from Bakardjian System.
        """
        ### 1. Firstly, load the data and extract components; those are modules 1. and 2.
        ### from original Bakardjian System
        self.load_data(data)

        ### 2. Secondly, filter the data on two or three frequencies, depending on what
        ### classification type you focuse on.
        self.bank_of_filters()

        ### 3. Extract energy band variance
        self.variance_analyzer()

        ### 4. Smooth the data using Savitzky-GOlay 2nd order filter
        self.smoothing()

        ### 5. Integrate channels
        self.integrating()

        ### 6. Normalize the data
        self.normalize()

        ### 7. Output is a data attribute.

    def fit_transform(self,data):

        self.fit(data)
        return(self.data)

    def predict(self):

        """
        Built-in classifier.
        """

        X = self.data.squeeze()
        C,P = X.shape
        classified = np.zeros((P))

        if self.threeclass == False:
            for n in range(P):
                dict_classes = {X[0,n]:0, X[1,n]:1}
                val_max = np.max(X[:,n])
                classified[n] = dict_classes[val_max]

        if self.threeclass == True:
            for n in range(P):
                dict_classes = {X[0,n]:0, X[1,n]:1, X[2,n]:2}
                val_max = np.max(X[:,n])
                classified[n] = dict_classes[val_max]
        y = int(mode(classified)[0][0])

        self.decision = y


# if __name__ is "__main__":
# bs = BakardjianSystem(freq = 256,channels=[15,23,28],
#                       extract=True,
#                       threeclass=True,
#                      seconds =3)
# data = load_data_path('C:\\Users\\stakar\\Documents\\BakSys\\data\\SUBJ1\\SSVEP_14Hz_Trial1_SUBJ1.csv',
# ',')
# bs.fit(data)
# bs.predict()
# print(bs.decision)
# print(bs.fit_transform(data)[0])

# TODO: In bank of filters, change the way of creating threeclass
# data, so it does not create new dataset, but rather just add
# Z array to dataset
# TODO: Extraction of morphological features
# TODO: Error handling
