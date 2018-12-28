from BakSys.BakSys import BakardjianSystem as BakSys
from feat_extraction.dataset_manipulation import *
from feat_extraction.features_extractor import Chromosome
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

# For the sake of learning, I am using dataset created for BakSys. However,
# latter analysis will be done on specially prepared data.
from sklearn.model_selection import train_test_split

bs = BakSys()
data = load_dataset('data/dataset.npy')
data,target = chunking(data)
X = np.array([bs.fit_transform(data[n]) for n in range(900)]).reshape(900*3,256)
y = np.array([[n,n,n] for n in target]).reshape(900*3)
X = X[np.where(y!=2)]
y = y[np.where(y!=2)]
X_train,X_test,y_train,y_test = train_test_split(X,y)


class GenAlFeaturesSelector(object):

    def __init__(self,train_data,test_data,train_target,test_target,
                 n_features=10,n_population=10,n_genotype=17,
                 mutation_probability=0.02):

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
        self.population = np.zeros((n_population,n_genotype))

        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
        self.mlp = MLPClassifier()
        self.mutation_probability = mutation_probability

    @staticmethod
    def get_gene():
        """ Returns positive value """
        return 1

    def fit(self,data):
        """ Fits the data to model """
        #for each individuals
        for n in range(self.n_population):
            self.population[n][np.random.randint(0,self.n_genotype,
                               self.n_features)] = self.get_gene()

    def _check_fitness(self,genotype):
        """ Check the fitness of given individual, by creating a new dataset,
        that uses selected features, fitting it to the MultiLayerPreceptron
        and checking how well it performs on train part, returning accuracy"""
        #create Chromosome object, with genotype of individual
        chrom = Chromosome(genotype = genotype)
        #create train dataset that uses features of individual
        fen_train = np.array([chrom.fit_transform(self.train_data[n])
                             for n in range(self.train_data.shape[0])])

        #create test dataset that uses features of individual
        fen_test = np.array([chrom.fit_transform(self.test_data[n])
                             for n in range(self.test_data.shape[0])])
        #fit the neural network to the data
        self.mlp.fit(fen_train,self.train_target)
        #return fitness, i.e accuracy of model
        return self.mlp.score(fen_test,self.test_target)

    def _population_fitness(self,population):
        """ Checks the fitness for each individual in population, then returns
        it """
        return np.array([self._check_fitness(n) for n in population])


    @staticmethod
    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendant
        s, array of characters with shape [2,n_genotype] """
        n_heritage = np.random.randint(0,len(mother))
        child1 = np.concatenate([father[:n_heritage],
                                 mother[n_heritage:]])
        child2 = np.concatenate([mother[:n_heritage],
                                 father[n_heritage:]])
        return child1,child2

    def transform(self):
        """ Transform, i.e. execute an algorithm. """
        self.past_populations = self.population.copy()
        best_fitness = np.min(self._population_fitness(self.population))
        self.n_generation = 0
        for n in range(10):

        # For check, how does an algorithm performs, comment out line above,
        # and comment line below.

        # while best_fitness > self.desired_fitness:
            self.past_populations = np.vstack([self.past_populations,
                                               self.population])
            self.descendants_generation()
            self.random_mutation()
            self.n_generation += 1
            best_fitness = np.min(self._population_fitness(self.population))

    def descendants_generation(self):
        """ Selects the best individuals, then generates new population, with
        half made of parents (i.e. best individuals) and half children(descendan
        ts of parents) """
        #Two firsts individuals in descendants generation are the best individua
        #ls from previous generation
        pop_fit = self._population_fitness(self.population)
        self.population[:2] = self.population[np.argsort(pop_fit)][:2]
        #now,let's select best ones
        parents_pop = self.roulette()
        #Finally, we populate new generation by pairing randomly chosen best
        #individuals
        for n in range(2,self.n_population-1):
                father = parents_pop[np.random.randint(self.n_population)]
                mother = parents_pop[np.random.randint(self.n_population)]
                children = self._pairing(mother,father)
                self.population[(n)] = children[0]
                self.population[(n)+1] = children[1]

    def random_mutation(self):
        """ Randomly mutates the population, for each individual it checks wheth
        er to do it accordingly to given probability, and then generates new cha
        racter on random locus """
        population = self.population.copy()
        for n in range(self.n_population):
            decision = np.random.random()
            if decision < self.mutation_probability:
                which_gene = np.random.randint(self.n_genotype)
                population[n][which_gene] = self.get_gene()
        self.population = population

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_populatio
        n, low_individual_probability,high_individual_probability]"""
        max_val = 1
        pop_fitness = [max_val-n for n in
                       self._population_fitness(self.population)]
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
    def plot_fitness(self):
        """ It checks the mean fitness for each passed population and the fitnes
        s of best idividual, then plots it. """
        past_populations = self.past_populations.reshape(int(
        self.past_populations.shape[0]/self.n_population),
        self.n_population,self.n_genotype)
        N = past_populations.shape[0]
        t = np.linspace(0,N,N)
        past_fit_mean = [np.mean(self._population_fitness(past_populations[n]))
                                                          for n in range(N)]
        past_fit_max = [np.min(self._population_fitness(past_populations[n]))
                                                        for n in range(N)]
        plt.plot(t,past_fit_mean,label='population mean fitness')
        plt.plot(t,past_fit_max,label='population best individual\'s fitness')
        plt.xlabel('Number of generations')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ga = GenAlFeaturesSelector(X_train,X_test,y_train,y_test)
    data = load_dataset('data/dataset.npy')
    data,target = chunking(data)
    print(data.shape)
    print(data.shape)
    print(ga.population)
    ga.fit(data)
    print(ga.population)
    ga.transform()
    print(ga._population_fitness(ga.population))


"""
ToDo:

Apply genetic algorithm to the problem of feature selection

"""
