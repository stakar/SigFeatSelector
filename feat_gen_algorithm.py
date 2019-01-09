from BakSys.BakSys import BakardjianSystem as BakSys
from feat_extraction.dataset_manipulation import *
from feat_extraction.features_extractor import Chromosome
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pickle


class GenAlFeaturesSelector(object):

    def __init__(self,n_features=10,n_population=10,n_genotype=17,
                 mutation_probability=0.02,desired_fitness=0.9,kfold = 5):

        """
        Features selector that uses Genetic Algorithm.

        n_features : int
        number of features that are supposed to be extract

        n_population : int
        number of individuals in population


        n_genotype : int
        length of genotype, i.e. range of features among which we ca
        n select in fact, it is determined by maximal length of Chro
        mosome's attribute genotype.

        """

        self.self = self
        self.n_features = n_features
        self.n_population = n_population
        self.n_genotype = n_genotype
        self.mlp = MLPClassifier()
        self.mutation_probability = mutation_probability
        self.desired_fitness = desired_fitness

        self.hall_of_fame = []
        self.kfold = kfold

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

        target : array [n_samples,]

        target, i.e. which stimuli was presented for each sample

        Attributes
        ----------

        population : array
        first, randomly generated population

        val_ranges : tuple(array,array)

        ranges of data, used for cross validation

        """
        #Firstly, let's shuffle the data
        self.population = np.zeros((self.n_population,self.n_genotype))
        order = np.random.permutation(np.arange(data.shape[0]))
        self.data = data[order]
        self.target = target[order]

        #Creates a random population
        for n in range(self.n_population):
            self.population[n][np.random.randint(0,self.n_genotype,
                               self.n_features)] = self.get_gene()

        self.val_ranges = self.cvalidation_ranges()


    def cvalidation_ranges(self):
        """ Generates a ranges of chunks for cross validation """
        full_range = np.arange(0,self.data.shape[0])
        chunk = int(data.shape[0]/self.kfold)
        test_chunks = list()
        train_chunks = list()
        for n in range(self.kfold):
            test = np.arange(chunk*n,chunk*(n+1))
            train = np.delete(full_range,test)
            test_chunks.append(test)
            train_chunks.append(train)
        return train_chunks,test_chunks

    def fenotype(self,genotype):
        #create Chromosome object, with genotype of individual
        chrom = Chromosome(genotype = genotype)
        #create train dataset that uses features of individual
        fen_train = np.array([chrom.fit_transform(self.data[n])
                             for n in range(self.data.shape[0])])
        #crate a placeholder for classifiers
        classifiers = list()
        for train in self.val_ranges[0]:
            self.mlp.fit(self.data[train],self.target[train])
            classifiers.append(self.mlp)
        return classifiers


    def _check_fitness(self,fenotype):
        """ Check the fitness of given individual, by creating a new dataset,
        that uses selected features, fitting it to the MultiLayerPreceptron
        and checking how well it performs on train part, returning accuracy"""

        scores = np.zeros(self.kfold)
        for n in range(self.kfold):
            tmp = fenotype[n].score(self.data[self.val_ranges[1][n]],
                                    self.target[self.val_ranges[1][n]])
            # print(np.unique(self.target[self.val_ranges[1][n]]))
            scores[n] = tmp
            # print(tmp)
        return np.round(np.mean(scores),2)

    def _population_fitness(self,population):
        """ Checks the fitness for each individual in population, then returns
        it """
        self.population_fitness = np.array([self._check_fitness(n)
                                            for n in self.population_fenotype])
        return self.population_fitness


    @staticmethod
    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendant
        s, array of characters with shape [2,n_genotype] """
        n_heritage = np.random.randint(0,len(mother))
        child1 = np.concatenate([father[:n_heritage],mother[n_heritage:]])
        child2 = np.concatenate([mother[:n_heritage],father[n_heritage:]])
        return child1,child2

    def transform(self):
        """ Transform, i.e. execute an algorithm. """
        self.population_fenotype = [self.fenotype(n) for n in self.population]
        self.past_populations = self._population_fitness(self.population)
        self.best_fitness = np.max(self.past_populations)
        self.n_generation = 0
        for n in range(5):

        # For check, how does an algorithm performs, comment out line above,
        # and comment line below.

        # while self.best_fitness < self.desired_fitness:
            self.descendants_generation()
            self.random_mutation()
            self.n_generation += 1
            self.best_fitness = np.max(self._population_fitness(self.population))



    def descendants_generation(self):
        """ Selects the best individuals, then generates new population, with
        half made of parents (i.e. best individuals) and half children(descendan
        ts of parents) """
        #Two firsts individuals in descendants generation are the best individua
        #ls from previous generation
        pop_fit = self._population_fitness(self.population)
        self.past_populations = np.vstack([self.past_populations,pop_fit])
        #now,let's select best ones
        self.population[:2] = self.population[np.argsort(pop_fit)][:2]
        self.population_fenotype[0] = self.population_fenotype[np.argsort(pop_fit).tolist()[0]]
        self.population_fenotype[1] = self.population_fenotype[np.argsort(pop_fit).tolist()[1]]
        parents_pop = self.roulette()
        #Finally, we populate new generation by pairing randomly chosen best
        #individuals
        for n in range(2,self.n_population-1):
                father = parents_pop[np.random.randint(self.n_population)]
                mother = parents_pop[np.random.randint(self.n_population)]
                children = self._pairing(mother,father)
                self.population[(n)] = children[0]
                self.population[(n)+1] = children[1]
                self.population_fenotype[n] = self.fenotype(self.population[n])
                self.population_fenotype[n] = self.fenotype(self.population[n+1])

    def random_mutation(self):
        """ Randomly mutates the population, for each individual it checks wheth
        er to do it accordingly to given probability, and then generates new cha
        racter on random locus """
        population = self.population.copy()
        for n in range(self.n_population):
            decision = np.random.random()
            if decision < self.mutation_probability:
                which_gene = np.random.randint(self.n_genotype)
                if population[n][which_gene] == 0:
                    population[n][which_gene] = self.get_gene()
                else:
                    population[n][which_gene] = 1
        self.population = population

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_populatio
        n, low_individual_probability,high_individual_probability]"""
        # max_val = 1
        # pop_fitness = [max_val-n for n in
        #                self._population_fitness(self.population)]
        pop_fitness = self._population_fitness(self.population)
        wheel = np.zeros((self.n_population,3))
        prob = 0
        for n in range(self.n_population):
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
        return np.array([self.population[self.roulette_swing(wheel)]
                         for n in range(self.n_population)])

        # return np.array([self.population[self.roulette_swing(wheel)]]).squeeze()
    def plot_fitness(self):
        """ It checks the mean fitness for each passed population and the fitnes
        s of best idividual, then plots it. """
        # past_populations = self.past_populations.reshape(int(
        # self.past_populations.shape[0]/self.n_population),
        # self.n_population,self.n_genotype)
        N = self.past_populations.shape[0]
        t = np.linspace(0,N,N)
        past_fit_mean = [np.mean(self.past_populations[n]) for n in range(N)]
        past_fit_max = [np.max(self.past_populations[n]) for n in range(N)]
        plt.plot(t,past_fit_mean,label='population mean fitness')
        plt.plot(t,past_fit_max,label='population best individual\'s fitness')
        plt.xlabel('Number of generations')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    bs = BakSys()

    ga = GenAlFeaturesSelector()
    data = load_dataset('datasetSUBJ1.npy')

    data,target = chunking(data)
    n_samples = target.shape[0]
    data = np.array([bs.fit_transform(data[n])
                          for n in range(n_samples)]).reshape(n_samples*3,256)
    target = np.array([[n,n,n] for n in target]).reshape(n_samples*3)

    # data = load_dataset('data/dataset.npy')
    # data,target = chunking(data)
    # print(data.shape)
    print(data.shape)
    ga.fit(data,target)
    print(ga.population)
    fenotype = ga.fenotype(ga.population[0])
    print(ga._check_fitness(fenotype))
    print(ga.mlp)
    # ga.transform()
    # print(ga._population_fitness(ga.population))
    # ga.plot_fitness()


"""
ToDo:

self._check_fitness works differently for each individual than _population_fitness, fix it
"""
