import numpy as np
import matplotlib.pyplot as plt
from string import ascii_letters, punctuation

class GenAlgorithmString(object):

    def __init__(self,symbols = (ascii_letters + punctuation + ' '),
                 n_population=10,desired_fitness=0.4,
                 mutation_probability = 0.02):

        """
        This is implementation of genetic algorithm, prepared for MD seminary
        presentation, it tries to generate given input by simulating evolutio
        n mechanisms.

        I used sklearn convention in naming fundamental part of it, so by cal
        ling fit method one can fits the word/sentence and by calling transfo
        rm one can activates algorithm so it tries to find the best (most sim
        ilar) string.

        Attributes

        symbols
        a set of characters from which algorithm should have created string,
        by default it is a set of asii letters and punctuation signs

        n_population
        the quantity of individuals for each population

        n_generation
        in how many generations algorithm should have found the best individual

        desired_fitness
        threshold that must be achieved to stop algorithm

        """
        self = self
        self.symbols = symbols
        self.n_population = n_population
        self.desired_fitness = desired_fitness
        self.mutation_probability = mutation_probability


    def get_symbol(self):
        """ Generates random symbol """
        return self.symbols[np.random.randint(len(self.symbols))]

    def fit(self,aim):
        """ Fits the given sentence to the mod el. Input aim is the string, sent
        ence that the algorithm is supposed to found. Generates the population"""
        self.target = np.array([n for n in aim],dtype = 'U1')
        self.n_genotype = len(self.target)
        self.population = np.array([[self.get_symbol() for
            n in range(self.n_genotype)] for n in range(self.n_population)])

    def transform(self):
        """ Transform, i.e. execute an algorithm. """
        self.past_populations = self.population.copy()
        best_fitness = np.min(self._population_fitness(self.population))
        self.n_generation = 0
        # for n in range(1000):

        # For check, how does an algorithm performs, comment out line above,
        # and comment line below.

        while best_fitness > self.desired_fitness:
            self.past_populations = np.vstack([self.past_populations,
                                               self.population])
            self.descendants_generation()
            self.random_mutation()
            self.n_generation += 1
            best_fitness = np.min(self._population_fitness(self.population))

    def _check_fitness(self,chromosome,target):
        """ Checks the fitness of individual. Fitness is the mesure of distance
        between letter generated randomly and the one that maps it on the locus
        in target array"""
        return sum([abs(ord(chromosome[n])-ord(target[n])) for n in range(
                       self.n_genotype)])

    def _population_fitness(self,population):
        """Returns an arrray with fitness of each individual in population"""
        return np.array([self._check_fitness(n,self.target) for
                         n in population])

    @staticmethod
    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendants, array of
        characters with shape [2,n_genotype] """
        n_heritage = np.random.randint(0,len(mother))
        child1 = np.concatenate([father[:n_heritage],mother[n_heritage:]])
        child2 = np.concatenate([mother[:n_heritage],father[n_heritage:]])
        return child1,child2

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

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_populatio
        n, low_individual_probability,high_individual_probability]"""
        max_val = 126*self.n_genotype
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

    def random_mutation(self):
        """ Randomly mutates the population, for each individual it checks wheth
        er to do it accordingly to given probability, and then generates new cha
        racter on random locus """
        population = self.population.copy()
        for n in range(self.n_population):
            decision = np.random.random()
            if decision < self.mutation_probability:
                which_gene = np.random.randint(self.n_genotype)
                population[n][which_gene] = self.get_symbol()
        self.population = population

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

    def best_individual(self,population):
        return population[np.argmin(self._population_fitness(population))]



if __name__ == '__main__':
    genalg = GenAlgorithmString(n_population=10,
                                desired_fitness = 0,mutation_probability=0.02)
    genalg.fit('Code!')
    pop1 = genalg.population
    print(pop1)
    genalg.transform()
    print(genalg._population_fitness(genalg.population))

    print(genalg.best_individual(genalg.population))
    genalg.plot_fitness()
