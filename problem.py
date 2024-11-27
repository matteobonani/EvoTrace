from pymoo.core.problem import ElementwiseProblem
import numpy as np


class MyProblem(ElementwiseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population):


        # find the max value for each column
        # population = np.array(initial_population)
        # xu = np.max(population, axis=0)  # max value per column
        # xl = np.zeros(trace_length)

        super().__init__(n_var=trace_length,
                         n_obj=2,  # objectives: constraints and diversity
                         n_constr=0)

        self.trace_length = trace_length
        self.encoder = encoder
        self.d4py = d4py
        self.initial_population = initial_population
        self.current_population = self.initial_population

    def evaluate_constraints(self, trace):
        """
        Evaluate the constraint violations of the given trace.
        """
        # TODO convert the encoded trace back into event names

        # TODO convert trace into a format usable by Declare4Py
        logs=0

        # perform conformance checking
        results = self.d4py.conformance_checking(logs, consider_vacuity=True)

        # count violations for each constraint violated
        violation_score = 0
        for constraint, result in results.items():
            violation_score += result.get('num_violations', 0)

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
        # TODO constraint must be weighted and diversity should use the current population
        # objective 1: Constraint violations
        constraint_score = self.evaluate_constraints(x)

        # objective 2: Diversity
        diversity_score = -self.calculate_diversity(x, self.current_population)

        out["F"] = [constraint_score, diversity_score]




