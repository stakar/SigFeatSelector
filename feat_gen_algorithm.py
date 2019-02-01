from BakSys.BakSys import BakardjianSystem as BakSys
from feat_extraction.dataset_manipulation import *
from feat_extraction.features_extractor import Chromosome
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score




class GenAlFeaturesSelector(object):

    def __init__(self,n_feat=1,n_pop=10,n_gen=17,
                 mut_prob=0.02,desired_fit=0.6,max_gen = 300,
                 scaler = MinMaxScaler(),
                 clf = MLPClassifier(random_state=42,max_iter=800,
                                     tol=1e-3)):

        """
        Features selector that uses Genetic Algorithm.

        n_feat : int
        number of features that are supposed to be extract

        n_pop : int
        number of individuals in pop


        n_gen : int
        length of genotype, i.e. range of features among which we ca
        n select in fact, it is determined by maximal length of Chro
        mosome's attribute genotype.

        """

        self.self = self
        self.n_feat = n_feat
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.pop = np.zeros((n_pop,n_gen))
        self.scaler = scaler
        self.clf = clf
        self.mlp = make_pipeline(self.scaler,self.clf)
        # self.mlp = self.clf
        self.mut_prob = mut_prob
        self.desired_fit = desired_fit
        self.max_gen = max_gen

    @staticmethod
    def get_gene():
        """ Returns positive value """
        return 1

    def fit(self,data,target):
        """ Fits the data to model. Fristly shuffles data, then

        Parameters
        ----------

        data : array [n_samples,frequency*time_window]

        offline data, consists of samples, i.e. transformed by BakardjianSystem
        EEG signal

        """
        #Firstly, let's shuffle the data
        order = np.random.permutation(np.arange(data.shape[0]))
        self.data = data[order]
        self.target = target[order]

        #Creates a random pop
        for n in range(self.n_pop):
            self.pop[n][np.random.randint(0,self.n_gen,
                               self.n_feat)] = self.get_gene()

    def _check_fitness(self,genotype):
        """ Check the fitness of given individual, by creating a new dataset,
        that uses selected features, fitting it to the MultiLayerPreceptron
        and checking how well it performs on train part, returning accuracy"""
        #create Chromosome object, with genotype of individual
        if all(genotype == 0):
            return 0
        chrom = Chromosome(genotype = genotype)

        #create train dataset that uses features of individual
        fen_train = np.array([chrom.fit_transform(self.data[n])
                             for n in range(self.data.shape[0])],dtype='float64')
        #return fitness, i.e accuracy of model
        score = np.mean(cross_val_score(self.mlp,fen_train,self.target,cv=5))
        while score > 0.99:
            score = np.mean(cross_val_score(self.mlp,fen_train,self.target,cv=5))
        return round(score,2)


    def _pop_fitness(self,pop):
        """ Checks the fitness for each individual in pop, then returns
        it """
        return np.array([self._check_fitness(n) for n in pop])


    @staticmethod
    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendant
        s, array of characters with shape [2,n_gen] """
        n_heritage = np.random.randint(0,len(mother))
        child1 = np.concatenate([father[:n_heritage],mother[n_heritage:]])
        child2 = np.concatenate([mother[:n_heritage],father[n_heritage:]])
        return child1,child2

    def transform(self):
        """ Transform, i.e. execute an algorithm. """
        self.past_pop = self._pop_fitness(self.pop)
        self.best_ind = np.max(self.past_pop)
        self.n_generation = 0
        # for n in range(10):

        # For check, how does an algorithm performs, comment out line above,
        # and comment line below.

        while self.best_ind < self.desired_fit:
            self.descendants_generation()
            self.random_mutation()
            self.n_generation += 1
            if self.n_generation > self.max_gen:
                break


    def fit_transform(self,data,target):
        """ Fits the data to model, then executes an algorithm. """
        self.fit(data,target)
        self.transform()

    def descendants_generation(self):
        """ Selects the best individuals, then generates new pop, with
        half made of parents (i.e. best individuals) and half children(descendan
        ts of parents) """
        #Two firsts individuals in descendants generation are the best individua
        #ls from previous generation
        self.pop_fit = self._pop_fitness(self.pop)
        if (self.n_generation % 50) == 0:
            print(self.pop_fit)
        self.best_ind = np.max(self.pop_fit)
        self.past_pop = np.vstack([self.past_pop,self.pop_fit])
        self.pop[:2] = self.pop[np.argsort(self.pop_fit)][-2:]
        #now,let's select best ones
        # print(pop_fit)
        parents_pop = self.roulette()
        #Finally, we populate new generation by pairing randomly chosen best
        #individuals
        for n in range(2,self.n_pop-1):
                father = parents_pop[np.random.randint(self.n_pop)]
                mother = parents_pop[np.random.randint(self.n_pop)]
                children = self._pairing(mother,father)
                self.pop[(n)] = children[0]
                self.pop[(n)+1] = children[1]

    def random_mutation(self):
        """ Randomly mutates the pop, for each individual it checks wheth
        er to do it accordingly to given probability, and then generates new cha
        racter on random locus """
        pop = self.pop.copy()
        for n in range(self.n_pop):
            decision = np.random.random()
            if decision < self.mut_prob:
                which_gene = np.random.randint(self.n_gen)
                if pop[n][which_gene] == 0:
                    pop[n][which_gene] = self.get_gene()
                else:
                    pop[n][which_gene] = 0
        self.pop = pop

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_populatio
        n, low_individual_probability,high_individual_probability]"""
#         pop_fitness = self._pop_fitness(self.pop)
        pop_fitness = self.pop_fit
        wheel = np.zeros((self.n_pop,3))
        prob = 0
        for n in range(self.n_pop):
            ind_prob = prob + (pop_fitness[n] / np.sum(pop_fitness))
            wheel[n] = [n,prob,ind_prob]
            prob = ind_prob
        return wheel

    def roulette_swing(self,wheel):
        """ This method takes as an input roulette wheel and returns an index of
        randomly chosen field """
        which = np.random.random()
        for n in range(len(wheel)):
            if which > wheel[n][1] and which < wheel[n][2]:
                return int(wheel[n][0])

    def roulette(self):
        """ This method performs selection of individuals, it takes the coeffici
        ent k, which is number of new individuals """
        wheel = self.roulette_wheel()
        return np.array([self.pop[self.roulette_swing(wheel)]
                         for n in range(self.n_pop)])

    def plot_fitness(self,title='Algorithm performance'):
        """ It checks the mean fitness for each passed pop and the fitnes
        s of best idividual, then plots it. """
        N = self.past_pop.shape[0]
        t = np.linspace(0,N,N)
        past_fit_mean = [np.mean(self.past_pop[n]) for n in range(N)]
        past_fit_max = [np.max(self.past_pop[n]) for n in range(N)]
        plt.plot(t,past_fit_mean,label='pop mean fitness')
        plt.plot(t,past_fit_max,label='pop best individual\'s fitness')
        plt.xlabel('Number of generations')
        plt.ylabel('Fitness')
        plt.ylim([0,1])
        plt.legend()
        plt.title(title)
        plt.savefig(title)
        # plt.show()



if __name__ == '__main__':
    time_window=1
    bs = BakSys(threeclass=False,seconds=time_window)
    ga = GenAlFeaturesSelector(n_pop=5,desired_fit=0.8)
    data = load_dataset('datasetSUBJ1.npy')
    data,target = chunking(data,time_window=time_window)
    n_samples = target.shape[0]
    freq = 256
    data = np.array([bs.fit_transform(data[n])
                          for n in range(n_samples)]).reshape(n_samples*2,freq*time_window)
    target = np.array([[n,n] for n in target]).reshape(n_samples*2)
    print(data.shape)
    ga.fit(data,target)
    print(ga.pop)
    # ga.transform()

"""
ToDo:

"""
