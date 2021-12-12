"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
"""
Code below "# Manas Rajendran signifies external addition by Manas R

"""
from functools import reduce
from operator import add
import random
from network import Network
# Manas Rajendran
import numpy as np


class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        """Create an optimizer.

        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father, mutationType):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                if mutationType == 1:
                    network = self.mutate(network)
                elif mutationType == 2:
                    network = self.randomMutate(network)

            children.append(network)

        return children

    # Manas Rajendran
    def breedNoMutate(self, father, mother):
        children = []
        for _ in range(2):

            child = {}

            # loop through the parameters and pick params for the child
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # create network object
            network = Network(self.nn_param_choices)
            network.create_set(child)
            """
            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                if mutationType == 1:
                    network = self.mutate(network)
                elif mutationType == 2:
                    network = self.randomMutate(network)
            """
            children.append(network)
        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params.
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network

    # Manas Rajendran
    def randomMutate(self, network):
        """ This is the same method as mutate(self, network) rather with a random amount of parameters mutated instead
        of only one."""

        # Randomize number of mutations

        mutationNumber = random.randint(0, (len(list(self.nn_param_choices.keys()))))

        for x in mutationNumber:
            mutation = random.choice(list(self.nn_param_choices.keys()))

            network.network[mutation] = random.choice(self.nn_param_choices)

        return network

    # Manas Rajendran
    def elitismMutate(self, pop, mutationType):
        """Evolve by selection and mutation to fill the remaining spots."""

        # same as the general purpose evolve method

        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        cPreMutate = []

        # now to mutate existing parents to generate the children - this process allows for the exclusion of crossover

        # This is an incorrect implementation
        """
        while len(children) < desired_length:

            # pick a random parent
            parent = random.randint(0, parents_length-1)

            parent = parents[parent]

            # randomize mutation rate
            mutater = random.uniform(float(mutationRate*(-1)),float(mutationRate))

            child = parent + (parent * mutater)

            children.append(child)

        parents.extend(children)

        return parents
        
        """

        # children = [self.randomMutate(network) for network in parents]

        while len(cPreMutate) < desired_length:
            preMutate = random.randint(0, parents_length - 1)
            cPreMutate.append(parents[preMutate])

        if mutationType == 1:
            children = [self.mutate(network) for network in cPreMutate]
        else:
            children = [self.randomMutate(network) for network in cPreMutate]

        parents.extend(children)

        """
        while len(parents) < desired_length:
            parents.append(random.choice(children))
        """
        return parents

    def elitismCrossoverMutate(self, pop, mutationType):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female, mutationType)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

    # Manas Rajendran
    def rouletteMutate(self, pop, mutationType):
        """In this evolution configuration, the process will use roulette selection to select parents for the next generation and
        mutate without crossover to populate the remaining spaces for the generation.
        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        """
        --first attempt at coding a roulette selection algorithm--
        
        used numpy.random.choice(population, size= , replace= ,p=[]) 
        
        """
        totalFitness = sum([(self.fitness(network), network) for network in pop])

        network_probabilities = []

        for x in graded:
            network_probabilities.append(graded[x] / totalFitness)

        parents = np.random.choice(pop, size=retain_length, replace=False, p=network_probabilities)

        print(parents)

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Standard mutation after this point similar to preMutate_NoCross()

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        cPreMutate = []

        while len(cPreMutate) < desired_length:
            preMutate = random.randint(0, parents_length - 1)
            cPreMutate.append(parents[preMutate])

        if mutationType == 1:
            children = [self.mutate(network) for network in cPreMutate]
        else:
            children = [self.randomMutate(network) for network in cPreMutate]

        parents.extend(children)

        """
        while len(parents) < desired_length:
            parents.append(random.choice(children))
        """
        return parents

    # Manas Rajendran
    def rouletteCrossoverMutate(self, pop, mutationType):
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        totalFitness = sum([(self.fitness(network), network) for network in pop])

        network_probabilities = []

        for x in graded:
            network_probabilities.append(graded[x] / totalFitness)

        parents = np.random.choice(pop, size=retain_length, replace=False, p=network_probabilities)

        print(parents)

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Crossover

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female, mutationType)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

    # Manas Rajendran
    def elitismCrossover(self, pop):
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breedNoMutate(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

    # Manas Rajendran
    def rouletteCrossover(self, pop):
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        totalFitness = sum([(self.fitness(network), network) for network in pop])

        network_probabilities = []

        for x in graded:
            network_probabilities.append(graded[x] / totalFitness)

        parents = np.random.choice(pop, size=retain_length, replace=False, p=network_probabilities)

        print(parents)

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Crossover

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
