import numpy as np

class AMUSE:

    def __init__(self, x, n_sources, tau):

        """
        Class for performing AMUSE algorithm, for artifact rejection

        Source: http://dspandmath.blogspot.com/2015/12/blind-source-separation-with-python.html
        """

        self.x = x
        self.n_sources = n_sources
        self.tau = tau
        self.__calc()

    def __calc(self):

       #BSS using eigenvalue value decomposition

       #Program written by A. Cichocki and R. Szupiluk at MATLAB

        R, N = self.x.shape
        Rxx = np.cov(self.x)
        U, S, U = np.linalg.svd(Rxx)

        if R > self.n_sources:
            noise_var = np.sum(self.x[self.n_sources+1:R+1])/(R - (self.n_sources + 1) + 1)
        else:
            noise_var = 0

        h = U[:,0:self.n_sources]
        T = np.zeros((R, self.n_sources))

        for m in range(0, self.n_sources):
            T[:, m] = np.dot((S[m] - noise_var)**(-0.5) ,  h[:,m])

        T = T.T
        y = np.dot(T, self.x)
        R1, N1 = y.shape
        Ryy = np.dot(y ,  np.hstack((np.zeros((R1, self.tau)), y[:,0:N1 - self.tau])).T) / N1
        Ryy = (Ryy + Ryy.T)/2
        D, B  = np.linalg.eig(Ryy)

        self.W = np.dot(B.T, T)
        self.sources = np.dot(self.W, self.x)
