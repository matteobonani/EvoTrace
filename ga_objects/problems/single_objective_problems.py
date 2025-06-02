import numpy as np
from scipy.spatial.distance import pdist, cdist
from .base_problem import BaseElementWiseProblem, BaseProblem


class ProblemSingleElementWiseWise(BaseElementWiseProblem):
    """
    Single-objective problem with constraints.

    Objective:
    - Maximize diversity (minimize similarity to existing population)

    Constraints:
    - Constraint satisfaction score must be <= 0 for feasibility.
    """
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=1, n_constr=1)

    def _evaluate(self, x, out, *args, **kwargs):
        constraint_score = self.evaluate_constraints(x)
        # diversity_score = -self.calculate_diversity(x, self.current_population)  # pymoo minimizes, so diversity must be negative (high hamming distance = high diversity)

        # similarity_penalty = (1 + diversity_score / 50) * 30 # large penalty if diversity is low
        # diversity_score = diversity_score + similarity_penalty

        diversity_score = np.mean(cdist(self.current_population, x[None, :], metric='hamming'))

        out["G"] = [constraint_score]  # feasible if <= 0
        out["F"] = [-diversity_score]

class ProblemSingleElementWiseWiseNoConstraints(BaseElementWiseProblem):
    """
    Single-objective problem without constraints.

    Objective:
    - Maximize diversity (minimize similarity to existing population)
    """
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=1, n_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        diversity_score = -self.calculate_diversity(x, self.current_population) # pymoo minimizes, so diversity must be negative (high hamming distance = high diversity)
        out["F"] = [diversity_score]

class ProblemSingle(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=1, n_constr=1)

    def _evaluate(self, X, out, *args, **kwargs):
        constraint_scores = self.evaluate_constraints_batch(X)
        dist_matrix = cdist(X, self.current_population, metric='hamming')
        diversity_scores = np.mean(dist_matrix, axis=1)

        out["G"] = constraint_scores[:, None]
        out["F"] = -diversity_scores[:, None]