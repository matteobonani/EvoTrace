from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible


class GrowingNSGA2(NSGA2):

    def __init__(self,
                 pop_size=100,
                 survival_num=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament), #TODO  super
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        # Initialize the parent NSGA2 class
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival or RankAndCrowding(),  # Provide default survival if not given
            output=output,
            **kwargs)

        # Custom termination or other settings can be defined here
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.survival_num=survival_num

    def _advance(self, infills=None, n_survive=None, **kwargs):
        """
        Overrides the default _advance method in NSGA2 to control the number of individuals (traces)
        to survive, either based on the population size or other custom logic.

        """
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.survival_num, algorithm=self, **kwargs)


    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

class PreSurvivalNSGA2(NSGA2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize pre_survival_pop as an empty list
        self.pre_survival_pop = []

    def _advance(self, infills=None, **kwargs):
        self.pre_survival_pop = self.pop.copy()
        # Perform the standard NSGA-II behavior
        super()._advance(infills=infills, **kwargs)



