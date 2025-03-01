import time
from pymoo.core.termination import Termination


class MyTermination(Termination):
    def __init__(self, required_feasible_solutions, constraint_type, constraint_index=None, max_time=3600):
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
        super().__init__()
        self.required_feasible_solutions = required_feasible_solutions
        self.total_population = None
        self.constraint_type = constraint_type
        self.constraint_index = constraint_index
        self.max_time = max_time
        self.start_time = None  # Track optimization start time

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
            return 1.0  # Stop


        if feasible_count >= self.required_feasible_solutions:
            return 1.0  # Stop

        # normalize termination progress (ensuring it doesn’t reach 1.0 before criteria are met)
        return min(feasible_count / max(self.total_population or 1, 1), 0.99)
