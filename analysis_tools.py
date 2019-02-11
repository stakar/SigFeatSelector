#Let's start with importing necessary modules.
from BakSys.BakSys import BakardjianSystem as BakSys
from feat_gen_algorithm import GenAlFeaturesSelector
from feat_extraction.dataset_manipulation import *
from feat_extraction.features_extractor import Chromosome
from sklearn.preprocessing import *
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import math
import time


bs = BakSys()
def best_features(population,pop_fitness):
    cr = Chromosome()
    best_ind = population[np.argmax(pop_fitness)]
    descriptions = np.array(cr.genes_description)
    des = descriptions[np.where(best_ind == 1)]
    return des

    data = load_dataset(path)
    
def preprocessing(data,title='Subject data',freq=256,bak_sys = bs,n_class = 2,
                  time_window = 1):
    """ Loading preprocessed data """
    data,target = chunking(data,time_window=time_window)
    test_sample = data[0]
    n_samples = target.shape[0]
    data = np.array([bak_sys.fit_transform(data[n])
                          for n in range(n_samples)]).reshape(n_samples*n_class,freq*time_window)
    if n_class == 2:
        target = np.array([[n,n] for n in target]).reshape(n_samples*n_class)
    elif n_class == 3:
        target = np.array([[n,n,n] for n in target]).reshape(n_samples*n_class)
    
    return data,target,test_sample,title


def itr(bak_sys,probe,dataset,target,ind,clf,accuracy,number_of_commands = 2,
        time_window = 1):
    cr = Chromosome(ind)
    dataset = np.array([cr.fit_transform(n) for n in dataset])
    clf.fit(dataset,target)
    t = time.time()                   #First time measure
    uno = bs.fit_transform(probe)[0]   #BakSys extraction
    probe_trans = np.array(cr.fit_transform(uno.squeeze()))#feature extraction
    dec = list()
    
    clf.predict(probe_trans.reshape(1,-1))
#     if len(uno_feat) == 1:
#         uno_feat = uno_feat.reshape(1,-1)
#     else:
#         uno_feat = uno_feat.reshape(-1,1)
    t2 = time.time()                 #Second time measure
    time_performance = t2-t
    N = number_of_commands           # Number of commands
    #Succes accuracy
    P = accuracy
    #Time delay
    T = time_performance + time_window
    itr = (math.log2(N) + (P * math.log2(P)) + ((1-P)*math.log2((1-P)/(N-1))))/(T/60)
    return itr
