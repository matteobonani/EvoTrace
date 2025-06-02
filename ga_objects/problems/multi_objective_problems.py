import numpy as np
import pandas as pd
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from scipy.spatial.distance import cdist
from .base_problem import BaseElementWiseProblem, BaseProblem

class ProblemMultiElementWiseWise(BaseElementWiseProblem):
    """
    Multi-objective problem with constraints.

    Objectives:
    - Maximize diversity (minimize similarity to existing population)
    - Minimize constraint violations

    Constraints:
    - Constraint satisfaction score must be <= 0 for feasibility.
    """
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=2, n_constr=1)

    def evaluate_constraints(self, trace):
        """
        Evaluate constraints in a multi-objective setting.

        Parameters:
        ----------
        trace : list
            The encoded process trace.

        Returns:
        -------
        tuple
            (satisfy_score, violation_score)
        """
        decoded_trace = self.encoder.decode(trace)
        self.dataframe['concept:name'] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=False)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        metric_state = np.array(conf_check_res.get_metric(trace_id=0, metric="state"))
        metric_num_violation = np.array(conf_check_res.get_metric(trace_id=0, metric="num_violations"), dtype=np.float_)

        valid_values = metric_num_violation[np.isfinite(metric_num_violation)]
        violation_score = int(np.sum(valid_values))
        satisfy_score = np.sum(1 - metric_state)

        return satisfy_score, violation_score

    def _evaluate(self, x, out, *args, **kwargs):
        constraint_score, violation_score = self.evaluate_constraints(x)
        diversity_score = -self.calculate_diversity(x, self.current_population) # pymoo minimizes, so diversity must be negative (high hamming distance = high diversity)

        out["G"] = [constraint_score]  # feasible if <= 0
        out["F"] = [diversity_score, violation_score]

class ProblemMultiNoConstElementWiseWise(BaseElementWiseProblem):
    """
    Multi-objective problem without constraints.

    Objectives:
    - Maximize diversity (minimize similarity to existing population)
    - Minimize constraint violations
    """
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=2, n_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        constraint_score = self.evaluate_constraints(x)
        diversity_score = -self.calculate_diversity(x, self.current_population) # pymoo minimizes, so diversity must be negative (high hamming distance = high diversity)

        diversity_weight = 0.2
        constraint_weight = 0.8

        # normalize the diversity to range between 0 and 1
        max_diversity = 50
        normalized_diversity = diversity_score / max_diversity

        # normalize the constraint evaluation to range between 0 and 1
        max_constraint = 10
        normalized_constraint = constraint_score / max_constraint

        weighted_diversity = diversity_weight * normalized_diversity
        weighted_constraint_eval = constraint_weight * normalized_constraint

        out["F"] = [weighted_diversity, weighted_constraint_eval]

class ProblemMulti(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=2, n_constr=0)
        self.n_constraint = len(d4py.get_decl_model_constraints())



    def _evaluate(self, X, out, *args, **kwargs):
        constraint_scores = self.evaluate_constraints_batch(X)
        dist_matrix = cdist(X, self.current_population, metric='hamming')
        diversity_scores = np.mean(dist_matrix, axis=1)

        # normalization

        max_constraint = self.n_constraint

        normalized_diversity = diversity_scores
        normalized_constraints = constraint_scores / max_constraint

        out["F"] = [-normalized_diversity[:, None], normalized_constraints[:, None]]

class ProblemMultiObjectiveNovelty(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=3, n_constr=1)

        # novelty archive (empty at start)
        self.novelty_archive = np.empty((0, trace_length))

    def _evaluate(self, X, out, *args, **kwargs):

        constraint_scores = self.evaluate_constraints_batch(X)
        dist_matrix = cdist(X, self.current_population, metric='hamming')
        diversity_scores = np.mean(dist_matrix, axis=1)


        if self.novelty_archive.shape[0] > 0:
            novelty_matrix = cdist(X, self.novelty_archive, metric='hamming')

        else:
            novelty_matrix = cdist(X, self.current_population, metric='hamming')

        # k-nearest neighbors to compute novelty
        k = min(5, novelty_matrix.shape[1])
        novelty_scores = np.mean(np.sort(novelty_matrix, axis=1)[:, :k], axis=1)

        out["G"] = constraint_scores[:, None]
        out["F"] = np.column_stack([
            -diversity_scores,
            -novelty_scores
        ])

        top_n = 5
        top_n_idx = np.argsort(novelty_scores)[-top_n:]
        new_novel = X[top_n_idx]

        self.novelty_archive = np.vstack([self.novelty_archive, new_novel])

        # limit archive size
        archive_limit = 500
        if self.novelty_archive.shape[0] > archive_limit:
            self.novelty_archive = self.novelty_archive[-archive_limit:]