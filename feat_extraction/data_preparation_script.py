import numpy as np
import os
import re
import platform

class CreateDataset():

    def __init__(self,target = {"8":0,"14":1,"28":2},freq=256,
                 subject = 'SUBJ1',target_file_name="dataset.npy"):

        """
        Class for dataset creation. Searches for path, reads the  data and  then
        saves the data in npy format.

        Attributes
        ----------

        target : dictionary {string:integer}
        dictionary containing frequencies in form of string (how they look alike
        in names of files) and how they are supposed to be mapped in dataset (in
        teger)

        target_file_name : string
        name of file under which the dataset is supposed to be saved

        freq : integer
        frequency of signal

        """

        self = self
        self._target = target
        self.freq = freq
        self.subject = subject
        self.target_file_name = target_file_name


    def go_path(self):
        """
        Changes the directory for the one that contains data.
        """
        path = input('Insert path to data catalog \n')
        os.chdir(path)

    def _create_array(self,path):
        """
        Serves for creating an array from path. In name of path is supposed to
        be an information about targeted frequency, which is extracted and added
        in form of additional column.

        Parameters
        ----------

        path : string
        path to the file in which is the data to which array is supposed to fits
        """
        #load data
        x = np.loadtxt(path,delimiter = ',')
        #cut unnecessary parts (the stimuli was present 5-20 seconds)
        x = x[5*self.freq:20*self.freq]

        re_sult = re.search('(?P<freq>\d+)Hz',path)
        f = re_sult.group('freq')
        target = self._target[f]
        t = [target for n in range(x.shape[0])]
        return np.column_stack((x,t))


    def create_placeholder_array(self):
        """
        This method creates a placeholder in attribute data, so one can vertical
        ly stack the data forwards. The shape of output data is n_channels + 1
        (number of channels and 1 column for target).
        """
        X = np.loadtxt(os.listdir()[0],delimiter=',')
        _x = X.shape[1] + 1
        self.data = np.zeros(_x)

    def read_write(self):
        """
        This method iterates through all directories with data, reads it then
        stack every each to data attirbute.
        """
        X = self.data
        #For each directory in data parent directory open it, read csv files and
        #add to the dataset
        for key in self._target.keys():
            dir_list = [n for n in os.listdir() if '_{}'.format(key) in n]
            dir_list = [n for n in dir_list if self.subject in n]
            for z in dir_list:
                print(z)
                t = self._create_array(z)
                X = np.vstack((X,t))
        X = X[1:]
        self.data = X

    def run(self):
        """
        This method runs all modules of this class, then saves the data attribut
        e as dataset.npz.

        """
        self.go_path()
        self.create_placeholder_array()
        self.read_write()
        np.save(self.target_file_name,self.data)


if __name__ == "__main__":
    for n in range(1,5):
        dc = CreateDataset(target = {'8':0,'14':1},subject='SUBJ{}'.format(n),
                           target_file_name='datasetSUBJ{}.npy'.format(n))
        dc.run()
