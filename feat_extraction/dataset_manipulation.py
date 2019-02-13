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
        n_samples,n_channels,freq*time_window """
    target = dataset[:,-1:]
    data = dataset[:,:-1]
    n_iter = int(data.shape[0]/(freq*time_window))
# line below creates dataset with shape n_samples (i.e. number of iterations,
# number of instances), n_channels (number of channels, un data I am woriking
# with it is 128 active electrodes, and freq*time_window (for each instance n
# umber of seconds and samples for each seconds))
    len_data = freq*time_window
    data = np.array([data[(n*len_data):(n*len_data)+len_data,:]
                     for n in range(n_iter)]).reshape(n_iter,128,len_data)
    target = np.array([target[n*len_data] for n in range(n_iter)])
    return data,target
