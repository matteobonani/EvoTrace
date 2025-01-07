from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.termination import Termination


class MyTermination(Termination):
    def __init__(self, n_required):
        super().__init__()
        self.n_required = n_required

    def _update(self, algorithm):
        G = algorithm.pop.get("G")

        n_feasible = (G <= 0).sum()

        if n_feasible >= self.n_required:
            return 1.0  # termination condition met
        else:
            # Progress percentage
            return n_feasible / self.n_required