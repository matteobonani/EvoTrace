from pymoo.algorithms.moo.nsga2 import NSGA2
from scipy.spatial.distance import cdist
import numpy as np

class CustomNSGA2(NSGA2):
    def _infill(self):
        """
        Overriding the infill method to access population and offspring
        before selection, calculate diversity, and reassign diversity to F.
        """
        # Get the current population and the newly generated offspring
        pop = self.pop  # Current population
        offspring = self.off  # Newly generated offspring

        # Extract the decision variables (X) and fitness values (F)
        pop_X = pop.get("X")
        pop_F = pop.get("F")  # Fitness values of the current population

        off_X = offspring.get("X")
        off_F = offspring.get("F")  # Fitness values of the offspring

        # Combine the population and offspring decision variables (X)
        all_X = np.vstack([pop_X, off_X])

        # Calculate diversity using pairwise distances on the decision variables (X)
        dist_matrix = cdist(all_X, all_X, metric='hamming')  # You can use 'euclidean' for continuous variables
        n = dist_matrix.shape[0]
        diversity = (np.sum(dist_matrix, axis=1) - np.diag(dist_matrix)) / (n - 1)

        # Calculate the mean diversity
        # diversity = dist_matrix.mean(axis=1)
        mean_diversity = diversity.mean()

        # Print or store the diversity score (optional)
        print(f"Mean Diversity (before selection): {mean_diversity}")

        # Assign the diversity values to the `F` of both population and offspring
        # Update population `F`
        pop_size = pop_X.shape[0]
        pop.set("F", np.repeat(-diversity[:pop_size].reshape(-1, 1), pop_F.shape[1], axis=1))

        print(f"Diversity pop (before selection): {np.repeat(-diversity[:pop_size].reshape(-1, 1), pop_F.shape[1], axis=1).mean()}")

        # Update offspring `F`
        off_size = off_X.shape[0]
        offspring.set("F", np.repeat(-diversity[pop_size:].reshape(-1, 1), off_F.shape[1], axis=1))

        print(
            f"Diversity off (before selection): {np.repeat(-diversity[pop_size:].reshape(-1, 1), off_F.shape[1], axis=1).mean()}")

        # Continue with the standard infill procedure (call parent method)
        return super()._infill()