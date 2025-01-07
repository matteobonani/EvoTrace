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
        # Convert the feature range to a numpy array for easier handling
        feature_range = np.array(self.feature_range)  # Shape: (num_features, 2)

        # Generate a random matrix where each entry is a random float between 0 and 1
        mutation_mask = np.random.rand(*X.shape) < self.mutation_probability


        lower_bounds = feature_range[:, 0].reshape(1, -1)  # Shape (1, num_features)
        upper_bounds = feature_range[:, 1].reshape(1, -1)  # Shape (1, num_features)

        # Vectorized generation of random integers within the specified bounds
        random_values = np.random.randint(lower_bounds, upper_bounds + 1, size=X.shape)

        # Apply mutations: Replace the features with the random values where mutation_mask is True
        X[mutation_mask] = random_values[mutation_mask]

        return X

def test_mutation():

    population = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [10, 14, 15, 16],
                           [9, 18, 19, 20]])

    print("Original Population:")
    print(population)

    feature_range = [(1, 10), (2, 20), (3, 30), (4, 40)]

    mutation_probability = 0.5
    mutation = MyMutation(feature_range=feature_range, mutation_probability=mutation_probability)

    mutated_population = mutation._do(None, population)

    print("\nMutated Population:")
    print(mutated_population)


if __name__ == "__main__":
    test_mutation()