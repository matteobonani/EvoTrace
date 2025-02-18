from pymoo.core.termination import Termination


class MyTermination(Termination):
    def __init__(self, n_required):
        super().__init__()
        self.n_required = n_required  # number of feasible solutions required
        self.n_total = None  # track total population size

    def _update(self, algorithm):
        G = algorithm.pop.get("G")
        F = algorithm.pop.get("F")


        if G is None or len(G) == 0:
            return 0.0  # no solutions available, keep running

        n_feasible = (G <= 0).sum()



        if self.n_total is None:
            self.n_total = len(G)  # store initial population size

        if n_feasible >= self.n_required:
            return 1.0  # terminate when enough feasible solutions are found

        return min(n_feasible / max(self.n_total, 1), 0.99)
