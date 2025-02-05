from pymoo.core.mutation import Mutation
import numpy as np
import random

from pymoo.operators.repair.to_bound import set_to_bounds_if_outside


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

        feature_range = np.array(self.feature_range)  # shape: (num_features, 2)

        # generate a random matrix where each entry is a random float between 0 and 1
        mutation_mask = np.random.rand(*X.shape) < self.mutation_probability


        lower_bounds = feature_range[:, 0].reshape(1, -1)  # shape (1, num_features)
        upper_bounds = feature_range[:, 1].reshape(1, -1)  # shape (1, num_features)

        # vectorized generation of random integers within the specified bounds
        random_values = np.random.randint(lower_bounds, upper_bounds + 1, size=X.shape)

        # apply mutations: Replace the features with the random values where mutation_mask is True
        X[mutation_mask] = random_values[mutation_mask]

        return X

def mut_pm_int(X, xl, xu, eta, prob, at_least_once):
    n, n_var = X.shape
    assert len(eta) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = np.random.rand(n, n_var) < prob[:, None]
    if at_least_once:
        force_mutate = np.random.randint(0, n_var, n)
        for i in range(n):
            mut[i, force_mutate[i]] = True

    mut[:, xl == xu] = False
    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = np.random.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values (rounded to nearest integer)
    _Y = np.round(X + deltaq * (_xu - _xl)).astype(int)


    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    Xp[mut] = _Y


    Xp = set_to_bounds_if_outside(Xp, xl, xu).astype(int)

    return Xp

class IntegerPolynomialMutation(Mutation):

    def __init__(self, prob=0.9, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = eta  # corrected to match integer mutation

    def _do(self, problem, X, **kwargs):
        X = X.astype(int)  # ensure input is integer

        eta = np.full(len(X), self.eta)
        prob_var = self.get_prob_var(problem, size=len(X))

        Xp = mut_pm_int(X, problem.xl.astype(int), problem.xu.astype(int), eta, prob_var, at_least_once=self.at_least_once)

        return Xp


class IntegerMutation(Mutation):
    def __init__(self, prob=0.1, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once

    def _do(self, problem, X, **kwargs):
        Xp = X.copy()

        # Mutation mask (decides which values to mutate)
        mut_mask = np.random.rand(*X.shape) < self.prob

        # Ensure at least one mutation per individual if required
        if self.at_least_once:
            force_mutate = np.random.randint(0, X.shape[1], X.shape[0])
            for i in range(X.shape[0]):
                mut_mask[i, force_mutate[i]] = True

        # Generate mutation values (-1 or +1)
        mutation_values = np.random.choice([-1, 1], size=X.shape)

        # Apply mutation only where mut_mask is True
        Xp[mut_mask] += mutation_values[mut_mask]

        # Ensure mutated values stay within bounds
        Xp = np.clip(Xp, problem.xl, problem.xu)
        Xp = set_to_bounds_if_outside(Xp, problem.xl, problem.xu)

        return Xp




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