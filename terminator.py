from pymoo.core.termination import Termination

class MyTermination(Termination):
    def __init__(self, min_feasible_solutions):
        super().__init__()
        self.min_feasible_solutions = min_feasible_solutions

    def _do_continue(self, algorithm):

        pop = algorithm.pop

        # Count how many traces have G == 0 (feasible solutions with no violations)
        feasible_count = sum(individual.G == 0 for individual in pop)
        for individual in pop:
            print(f"Individual: {individual}, G: {individual.G}, G[0]: {individual.G[0]}")
        print(feasible_count)


        return feasible_count < self.min_feasible_solutions


    def _update(self, algorithm):

        return algorithm.n_gen