import time
from pymoo.core.termination import Termination
import numpy as np

class ConstraintTermination(Termination):
    """
    This class defines a stopping condition based on two criteria:
    (1) the optimization process terminates when a specified number of feasible solutions are found.
    (2) it stops if the maximum allowed runtime is exceeded.

    Parameters
    ----------
    required_feasible_solutions : int
        Number of feasible solutions required to terminate.

    constraint_type : str
        Specifies where the constraint values are stored for feasibility evaluation.
        Use "G" if constraint violations are recorded in 'G', or "F" if constraints should be evaluated based on objective values in 'F'.

    constraint_index : int
        The index in 'F' to check feasibility when constraint_type is "F".

    max_time : int
        Maximum runtime in seconds before stopping (default: 3600s = 1 hour).
    """
    def __init__(self, required_feasible_solutions, constraint_type, constraint_index=None, max_time=3600):

        super().__init__()
        self.required_feasible_solutions = required_feasible_solutions
        self.total_population = None
        self.constraint_type = constraint_type
        self.constraint_index = constraint_index
        self.max_time = max_time
        self.start_time = None

    def has_terminated(self):
        """Determine the time elapsed since the algorithm started."""
        if self.start_time is None:  # This runs only once
            self.start_time = time.time()
        return super().has_terminated()

    def _update(self, algorithm):
        """Checks termination condition based on the number of feasible solutions found OR time elapsed."""
        population = algorithm.pop

        # get constraints or objectives based on constraint_type
        if self.constraint_type == "G":
            constraint_values = population.get("G")
            feasible_count = (constraint_values <= 0).sum()
        else:
            objective_values = population.get("F")
            feasible_count = (objective_values[:, self.constraint_index] <= 0).sum()

        # check elapsed time
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_time:
            print(f"Termination: Max time of {self.max_time} seconds reached!")
            return 1.0

        if feasible_count >= self.required_feasible_solutions:
            return 1.0

        # normalize termination progress (ensuring it doesn’t reach 1.0 before criteria are met)
        return min(feasible_count / max(self.total_population or 1, 1), 0.99)

class DiversityTermination(Termination):
    """
    This class defines a stopping condition based on the achieved diversity score:
    (1) The optimization process terminates when a specified diversity threshold is reached.
    (2) It stops if the maximum allowed runtime is exceeded.

    Parameters
    ----------
    required_diversity_threshold : float
        The required diversity threshold value to stop the optimization.(from 0 to 1)

    max_time : int, optional
        Maximum runtime in seconds before stopping (default: 3600s = 1 hour).
    """

    def __init__(self, required_diversity_threshold: float, max_time=3600):
        super().__init__()
        self.required_diversity_threshold = required_diversity_threshold
        self.max_time = max_time
        self.start_time = None

    def _update(self, algorithm):
        if self.start_time is None:
            self.start_time = time.time()

        # get average diversity (negative so we negate it)
        F = algorithm.pop.get("F")[:, 0]
        average_diversity_score = -np.mean(F)

        if average_diversity_score >= self.required_diversity_threshold:
            # print(f"[Termination] Diversity threshold {self.required_diversity_threshold} reached.")
            return 1.0

        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_time:
            # print(f"[Termination] Max time {self.max_time:.1f}s reached.")
            return 1.0

        return min(self.required_diversity_threshold, 0.99)