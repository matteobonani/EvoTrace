import numpy as np
import random
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SBX


class ConstraintAwareCrossover(SBX):
    def _do(self, problem, X, **kwargs):
        max_retries = 10
        parents = X

        for _ in range(max_retries):
            offspring = super()._do(problem, parents, **kwargs)

            # check feasibility for each offspring
            feasible_offspring = []
            for i in range(offspring.shape[0]):  # Iterate over offspring
                child = offspring[i]

                constraint_values = problem.evaluate_constraints(child.reshape(1, -1))  # Ensure correct shape
                if np.all(constraint_values <= 0):  # if all constraints are satisfied
                    feasible_offspring.append(child)

            if len(feasible_offspring) > 0:  # if at least one valid solution was found, return them
                return np.array(feasible_offspring)

        return offspring  # if no feasible solutions were found after retries, return best attempt


class TraceCrossover(Crossover):
    def __init__(self, variable_boundaries):
        super().__init__(n_parents=2, n_offsprings=2)  # two parents produce two offspring
        self.variable_boundaries = variable_boundaries

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        Y = np.zeros_like(X)

        for k in range(n_matings):

            # get two parents
            parent1 = X[0, k, :]
            parent2 = X[1, k, :]

            # choose a random crossover point respecting boundaries
            crossover_idx = random.randint(1, len(self.variable_boundaries) - 2)
            crossover_point = self.variable_boundaries[crossover_idx]

            # generate offspring by swapping parts at the crossover point
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

            Y[0, k, :] = child1
            Y[1, k, :] = child2

        return Y


class UniformCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)

    def _do(self, problem, X, **kwargs):
        # X: shape = (n_parents, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape
        n_offsprings = self.n_offsprings

        # output shape: (n_offsprings, n_matings, n_var)
        off = np.empty((n_offsprings, n_matings, n_var), dtype=X.dtype)

        for i in range(n_matings):
            p1 = X[0, i]
            p2 = X[1, i]

            mask = np.random.rand(n_var) < 0.5
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)

            off[0, i] = c1
            off[1, i] = c2

        return off