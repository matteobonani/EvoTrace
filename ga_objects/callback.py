from pymoo.core.callback import Callback
import numpy as np


class UpdatePopulationCallback(Callback):
    """
    A callback class for tracking the optimization process and updating the current population.

    This class collects constraint violations, objective values, and generation counts
    to monitor the progress of the optimization run.
    """
    def __init__(self):
        super().__init__()

        self.constraint_history = []
        self.second_objective = []
        self.first_objective = []
        self.generation = 0


    def notify(self, algorithm):
        """Update the problem's current population and gather data for plotting."""

        population = algorithm.pop.get("X")

        algorithm.problem.set_current_population(population)

        # gather fitness and constraint scores for the current population
        G = algorithm.pop.get("G")
        if G is not None and G.size > 0:
            constraint_scores = G[:, 0]  # assuming G[0] corresponds to constraint violations
            self.constraint_history.append(np.mean(constraint_scores))
        F = algorithm.pop.get("F")
        first_objective = F[:, 0]  # assuming F[0] is the diversity score
        if F.shape[1] > 1:  # if G[1] exists (multi obj GA)
            second_objective = F[:, 1]
            self.second_objective.append(np.mean(second_objective))

        self.first_objective.append(np.mean(first_objective))


        self.generation += 1


    def get_data(self):
        """Retrieve and return recorded data from the optimization process."""
        return {
            "constraint_history": self.constraint_history,
            "second_objective": self.second_objective,
            "first_objective": self.first_objective,
            "generations": self.generation,
        }





