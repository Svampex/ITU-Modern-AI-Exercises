import numpy as np
import random
class Gene():
    def __init__(self, gene_seed):
        if isinstance(gene_seed, int):
            self.genome = np.random.binomial(1, 0.5, gene_seed)
        else:
            self.genome = np.copy(gene_seed)

        self.gene_size = len(self.genome)

    def mutate(self):
        i = 0
        while i < 3:
            self.genome[random.randint(self.gene_size)] = random.randint(2)
            i += 1

    def evaluate_fitness(self, target):
        """ Lower fitness is better. Perfect fitness should equal 0"""
        return np.linalg.norm(self.genome - target)


class GeneticAlgorithm():
    def __init__(self, gene_size, population_size, target):
        self.gene_size = gene_size
        self.pop_size = population_size
        self.target = target
        self.parents = []

        self.gene_pool = [[np.nan, Gene(self.gene_size)] for _ in range(self.pop_size )]

    def evaluate_population(self):
        """ Evaluates the fitness of ever genome. If the best fitness is 0
            the function returns True, signalling that the optimization is done.
        """
        min_fitness = np.inf
        for gene in self.gene_pool:
            fitness = gene[1].evaluate_fitness(self.target)
            if min_fitness > fitness:
                min_fitness = fitness
            gene[0] = fitness

        if min_fitness == 0:
            return True
        return False

    def select_parents(self, num_parents):
        """ Function that selects num_parents from the population."""
        candidates = np.random.choice(self.gene_pool, num_parents*2)
        winners = []
        while len(candidates) > 0:
            p1 = candidates[0]
            p2 = candidates[1]
            if p1[0] > p2[0]:
                winners.append(p1)
            else:
                winners.append(p1)
            candidates = np.delete(candidates, [0,1])

        self.parents = winners

    def produce_next_generation(self):
        """ Function that creates the next generation based on parents."""
        for i in range(len(self.parents)/2):
            p1gene =  self.parents[i*2][0:5]
            p2gene = self.parents[i*2+1][0:5]
            self.parents[i*2][0:5] = p2gene
            self.parents[i * 2+1][0:5] = p1gene
            # "Creates" new children (2 new, 2 parents die (just modifies parents in place into children))


    def run(self):
        done = False
        i = 1
        while not done :
            done = self.evaluate_population()
            self.select_parents(int(self.pop_size/10)+1)

            if i % 5 == 0 or done:
                print("Generation:", i)
                print("Population:")
                for gene in self.gene_pool:
                    print("\tfit:", gene[0], gene[1].genome)
                print("Parents:")
                for parent in self.parents:
                    print("\tfit:", parent[0], parent[1].genome)
                print()

            self.produce_next_generation()
            i += 1

pop_size = 10
gene_size = 20
target = np.zeros(gene_size)

GA = GeneticAlgorithm(gene_size, pop_size, target)
GA.run()

print('Done')