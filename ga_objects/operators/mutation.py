from pymoo.core.mutation import Mutation
import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside


# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------

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

    # back in bounds if necessary
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    # in case out of bounds repair (very unlikely)
    Xp = set_to_bounds_if_outside(Xp, xl, xu).astype(int)

    return Xp


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------

class IntegerPolynomialMutation(Mutation):
    """
    Integer version of polynomial mutation.

    This mutation operator perturbs integer-valued decision variables using
    polynomial mutation, ensuring that values remain within given bounds.

    Parameters:
    ----------
    prob : float, optional (default=0.9)
        The probability of mutation for each variable.

    eta : float, optional (default=20)
        The distribution index controlling mutation spread.

    at_least_once : bool, optional (default=False)
        If True, ensures at least one mutation occurs per individual.
    """

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

class RandomIntegerMutation(Mutation):
    """
    Random integer mutation.

    This mutation operator replaces integer-valued decision variables with
    random values selected uniformly from the allowed range.

    Parameters:
    ----------
    prob : float, optional (default=1.0)
        The probability of mutation for each variable.
    """
    def __init__(self, prob=1.0):
        super().__init__(prob=prob)

    def _do(self, problem, X, **kwargs):
        # Ensure input is integer
        Xp = X.astype(int)

        # Mutate by randomly choosing a value from bounds
        for i in range(X.shape[0]):  # iterate over population
            for j in range(X.shape[1]):  # iterate over variables
                if np.random.rand() < self.prob.value:  # mutation probability
                    # Randomly pick a new value from bounds xl, xu
                    Xp[i, j] = np.random.randint(problem.xl[j], problem.xu[j] + 1)

        return Xp


def main():

    class MyIntegerProblem(Problem):
        def __init__(self):
            super().__init__(n_var=3,
                             n_obj=1,
                             n_constr=0,
                             xl=np.array([0, 0, 0]),
                             xu=np.array([10, 10, 10]))
    X = np.array([[5, 6, 7], [2, 8, 3], [1, 4, 6]])

    problem = MyIntegerProblem()

    mutation = RandomIntegerMutation(prob=1.0)

    Xp = mutation._do(problem, X)

    print("Original X:")
    print(X)
    print("\nMutated Xp:")
    print(Xp)

if __name__ == "__main__":
    main()