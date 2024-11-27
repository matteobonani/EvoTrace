from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.termination import get_termination
from pymoo.core.sampling import Sampling
import numpy as np
import random

class GrowingPopulationNSGA2(NSGA2):
    def __init__(self, pop_size, sampling, growth_rate, max_pop_size, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.growth_rate = growth_rate
        self.max_pop_size = max_pop_size

    def advance(self, infills=None, **kwargs):
        super().advance(infills=infills, **kwargs)

        # if the population hasn't reached the maximum size, grow it
        if len(self.pop) < self.max_pop_size:
            additional_size = min(self.growth_rate, self.max_pop_size - len(self.pop))
            print(f"Adding {additional_size} individuals")


            new_individuals = self.initialization.do(self.problem, n_samples=additional_size)

            self.evaluator.eval(self.problem, new_individuals)

            self.pop = Population.merge(self.pop, new_individuals)

        print(f"Current Population Size: {len(self.pop)}")

