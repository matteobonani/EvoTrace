from pymoo.core.callback import Callback


# TODO plot la media della fit e score vincoli
class UpdatePopulationCallback(Callback):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def notify(self, algorithm):
        """
        Update the problem's current population with the latest population.
        """
        population = algorithm.pop.get("X")
        self.problem.set_current_population(population)