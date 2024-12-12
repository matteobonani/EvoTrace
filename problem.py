from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from pymoo.core.problem import ElementwiseProblem
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
import numpy as np
import pandas as pd

class MyProblem(ElementwiseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl,xu, event_log, dataframe):


        # find the max value for each column
        # population = np.array(initial_population)
        # xu = np.max(population, axis=0)  # max value per column
        # xl = np.zeros(trace_length)

        super().__init__(n_var=trace_length,
                         n_obj=2,  # objectives: constraints and diversity TODO use 1 obj
                         n_constr=1, # 1 constraint (total violation score)
                         xl=xl,
                         xu=xu)

        self.trace_length = trace_length
        self.encoder = encoder
        self.d4py = d4py
        self.initial_population = initial_population
        self.current_population = self.initial_population
        self.event_log = event_log
        self.dataframe = dataframe

    def evaluate_constraints(self, trace):
        """
        Evaluate the constraint violations of the given trace.
        """


        # TODO uno con la media e uno boolean
        decoded_trace = self.encoder.decode(trace)

        self.dataframe['concept:name'] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=False)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        metric_state = conf_check_res.get_metric(trace_id=0, metric="state")


        violation_score = 0

        for x in metric_state:
            if x == 1:
                violation_score += 1

        # violation_score = 1 if any(x != 0 for x in metric_state) else 0


        return violation_score

    def calculate_diversity(self, trace, population):
        """
        Calculate diversity as the average Hamming Distance of the trace
        compared to all traces in the population.

        All differences are treated equally, regardless of magnitude (e.g., 0 vs 1 is the same as 0 vs 3).
        """

        diversity = 0
        for other_trace in population:
            diversity += self.hamming_distance(trace, other_trace)

        return diversity / len(population)


    def hamming_distance(self, trace1, trace2):
        """
        Compute the Hamming Distance between two traces.
        """
        return sum(el1 != el2 for el1, el2 in zip(trace1, trace2))

    def set_current_population(self, population):
        """
        Update the current population.
        """
        self.current_population = np.array(population)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a single trace.
        """

        # objective 1: Constraint violations
        constraint_score = self.evaluate_constraints(x)

        # objective 2: Diversity
        diversity_score = -self.calculate_diversity(x, self.current_population) # negative because pymoo minimize, choose the lower value


        out["G"] = [constraint_score]  # feasible if <= 0

        out["F"] = [constraint_score, diversity_score]






