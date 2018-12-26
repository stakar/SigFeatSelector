import numpy as np

def load_dataset(*argv):
    """ Loads dataset from given path, if it is not passed, then asked for it"""
    if len(argv)>0:
        path = argv[0]
    else:
        path = input('Please pass the path to file')
    return np.load(path)

def chunking(dataset,time_window=1,freq=256):
    """ Decompose dataset into smaller chunks. Shape of returned object is
        n_samples,freq*time_window,n_channels """
    target = dataset[:,-1:]
    data = dataset[:,:-1]
    n_iter = int(dataset.shape[0]/(freq*time_window))
    data = np.array([data[(n*freq):(n*freq)+freq,:] for n in range(n_iter)])
    target = np.array([target[n*256] for n in range(n_iter)])
    return data,target
