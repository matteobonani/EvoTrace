from pymoo.algorithms.moo.nsga2 import NSGA2
from scipy.spatial.distance import cdist
import numpy as np

class CustomNSGA2(NSGA2):
    def _infill(self):
        """
        Overriding the infill method to access population and offspring
        before selection, calculate diversity, and reassign diversity to F.
        """
        pop = self.pop
        offspring = self.off


        pop_X = pop.get("X")
        pop_F = pop.get("F")

        off_X = offspring.get("X")
        off_F = offspring.get("F")

        # combine the population and offspring decision variables (X)
        all_X = np.vstack([pop_X, off_X])


        dist_matrix = cdist(all_X, all_X, metric='hamming')
        n = dist_matrix.shape[0]
        diversity = (np.sum(dist_matrix, axis=1) - np.diag(dist_matrix)) / (n - 1)


        # diversity = dist_matrix.mean(axis=1)
        mean_diversity = diversity.mean()

        # sssign the diversity values to the `F` of both population and offspring

        pop_size = pop_X.shape[0]
        pop.set("F", np.repeat(-diversity[:pop_size].reshape(-1, 1), pop_F.shape[1], axis=1))

        off_size = off_X.shape[0]
        offspring.set("F", np.repeat(-diversity[pop_size:].reshape(-1, 1), off_F.shape[1], axis=1))

        return super()._infill()