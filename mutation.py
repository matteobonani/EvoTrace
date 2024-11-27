from pymoo.core.mutation import Mutation
import numpy as np
import random

class MyMutation(Mutation):
    # TODO the obj already has a prob variable
    def __init__(self, feature_range, mutation_probability=0.2):
        """
        Mutation class with a probability for applying mutations to each individual.
        """
        super().__init__()
        self.feature_range = feature_range  # bounds for mutation
        self.mutation_probability = mutation_probability

    def _do(self, problem, X, **kwargs):
        """
        Perform mutation on a percentage of the population based on mutation_probability.
        """
        # TODO better to not modify more than once a dummy variable, and also check that inside the range, only one variable can be set as 1 [check expanding pop]
        # over each individual
        for i in range(len(X)):

            child = X[i]
            # over each feature
            for j in range(len(child)):
                if np.random.rand() < self.mutation_probability:
                    lower_bound, upper_bound = self.feature_range[j]
                    child[j] = random.randint(lower_bound, upper_bound)
            X[i] = child

        return X
