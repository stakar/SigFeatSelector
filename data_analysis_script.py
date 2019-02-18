#Let's start with importing necessary modules.
from BakSys.BakSys import BakardjianSystem as BakSys
from feat_gen_algorithm import GenAlFeaturesSelector
from feat_extraction.dataset_manipulation import *
from feat_extraction.features_extractor import Chromosome
from analysis_tools import *
from sklearn.preprocessing import *
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import math
import time


def subroutine(subject,time_window,gafeat):
    gafeat.fit_transform(subject[0],subject[1])
    gafeat.plot_fitness('{} sec '.format(time_window) + subject[3])
    plt.clf()
    best_feat = pd.Series(best_features(gafeat.pop,gafeat.pop_fit))
    ind = gafeat.pop[np.argmax(gafeat.pop_fit)]
    inf_ratio = itr(bs,subject[2],subject[0],subject[1],ind,clf=clf,
                    accuracy = gafeat.best_ind,time_window=time_window)
    subject_results = {'Score':gafeat.best_ind,
                        'Num of Generations':gafeat.n_generation,
                        'Best features':best_feat.str.cat(sep=',')}
    subject_results['ITR'] = round(inf_ratio)
    return subject_results

#loading data

subj1_raw = load_dataset('datasetSUBJ1.npy')
subj2_raw = load_dataset('datasetSUBJ2.npy')
subj3_raw = load_dataset('datasetSUBJ3.npy')
subj4_raw = load_dataset('datasetSUBJ4.npy')

#Setting parameters

n_population = 5
desired_fitness = 0.95,
max_generation = 10000
clf = MLPClassifier(max_iter=200,random_state=42,tol=1e-2)
scaler = MinMaxScaler()

gafeat = GenAlFeaturesSelector(n_pop=n_population,max_gen=max_generation,
                               desired_fit=desired_fitness,
                               scaler = scaler,clf=clf)

#for each time window perform analysis

for t_window in range(1,5):
    time_window = t_window

    tmp_results = list()

    bs = BakSys(threeclass=False,seconds = time_window)

    subj1 = preprocessing(subj1_raw,'subject 1 data',256,bs,n_class = 2,
                          time_window = time_window)
    subj2 = preprocessing(subj2_raw,'subject 2 data',256,bs,n_class = 2,
                          time_window = time_window)
    subj3 = preprocessing(subj3_raw,'subject 3 data',256,bs,
                          time_window = time_window)
    subj4 = preprocessing(subj4_raw,'subject 4 data',256,bs,
                          time_window = time_window)
    overall = (np.vstack([subj1[0],subj2[0],subj3[0],subj4[0]]),
               np.hstack([subj1[1],subj2[1],subj3[1],subj4[1]]),
               subj1[2],'overall data')

     for n in [subj1,subj2,subj3,subj4,overall]:
    #for n in [subj1,subj2]:
        tmp = subroutine(n,time_window,gafeat)
        tmp_results.append(tmp)

    df_results = summary = pd.DataFrame(columns=list(tmp_results[0].keys()),
             index=['Subject 1','Subject 2','Subject 3','Subject 4','Overall'],
             # index = ['Subj1','Subj2'],
             data=tmp_results)

    df_results.to_csv('results_{}.csv'.format(t_window))
