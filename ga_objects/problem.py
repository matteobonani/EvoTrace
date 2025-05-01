from Declare4Py.D4PyEventLog import D4PyEventLog
from pm4py.algo.simulation.playout.process_tree.variants.extensive import flatten
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from pymoo.core.problem import Problem
from scipy.spatial.distance import pdist, cdist
from joblib import Parallel, delayed
import time



class BaseElementProblem(ElementwiseProblem):
    """
    Base class for defining element-wise optimization problems.

    This class sets up common components for problems that involve optimizing
    sequences (traces) by evaluating diversity and constraint satisfaction.

    Parameters:
    ----------
    trace_length : int
        The length of the traces.
    encoder : Encoder
        The encoder to convert between activity names and integer representations.
    d4py : DeclareForPyModel
        The declarative model used for constraint checking.
    initial_population : list
        The initial set of traces.
    xl : np.ndarray or int
        The lower bounds for decision variables.
    xu : np.ndarray or int
        The upper bounds for decision variables.
    event_log : EventLog
        The event log containing process execution traces.
    dataframe : pd.DataFrame
        The dataframe representation of the event log.
    n_obj : int
        The number of objectives in the optimization problem.
    n_constr : int
        The number of constraints in the optimization problem.
    """
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj, n_constr):
        super().__init__(n_var=trace_length, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.trace_length = trace_length
        self.encoder = encoder
        self.d4py = d4py
        self.initial_population = np.array(initial_population)
        self.current_population = self.initial_population
        self.event_log = event_log
        self.dataframe = dataframe

    def evaluate_constraints(self, trace):
        """
        Evaluate constraint violations for the given trace.

        Parameters:
        ----------
        trace : list
            The encoded process trace.

        Returns:
        -------
        int
            The total number of violated constraints.
        """
        decoded_trace = self.encoder.decode(trace)
        self.dataframe['concept:name'] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=False)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        metric_state = np.array(conf_check_res.get_metric(trace_id=0, metric="state"))
        metric_state_inverted = 1 - metric_state

        return np.sum(metric_state_inverted)

    def calculate_diversity(self, trace, population):
        """Calculate diversity as the average Hamming Distance of the trace."""
        return np.mean(np.sum(population != trace, axis=1))

    def set_current_population(self, population):
        """Update the current population."""
        self.current_population = np.array(population)


class ProblemMultiElementWise(BaseElementProblem):
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


class ProblemMultiNoConstElementWise(BaseElementProblem):
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


class ProblemSingleElementWise(BaseElementProblem):
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


class ProblemSingleElementWiseNoConstraints(BaseElementProblem):
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

class BaseProblem(Problem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj, n_constr):
        super().__init__(n_var=trace_length, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.trace_length = trace_length
        self.encoder = encoder
        self.d4py = d4py
        self.initial_population = np.array(initial_population)
        self.current_population = self.initial_population
        self.event_log = event_log
        self.dataframe = dataframe

    def evaluate_constraints_batch(self, X):
        return np.array(Parallel(n_jobs=-1)(delayed(self.evaluate_constraints)(trace) for trace in X))

    def evaluate_constraints(self, trace):
        decoded_trace = self.encoder.decode(trace)
        self.dataframe["concept:name"] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=True)
        conf_check_res = basic_checker.run()
        metric_state = np.array(conf_check_res.get_metric(trace_id=0, metric="state"))
        return np.sum(1 - metric_state)

    def evaluate_all_constraints(self, population):
        decoded_pop = self.encoder.decode(population)
        flattened_decoded_pop = np.array(decoded_pop).flatten()
        self.dataframe["concept:name"] = pd.DataFrame(flattened_decoded_pop)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=True)
        conf_check_res = basic_checker.run()
        metric_states = np.array(conf_check_res.get_metric(metric="state"))
        return np.sum(1 - metric_states, axis=1)

    def set_current_population(self, population):
        self.current_population = np.array(population)


class ProblemSingle(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=1, n_constr=1)

    def _evaluate(self, X, out, *args, **kwargs):
        constraint_scores = self.evaluate_constraints_batch(X)
        dist_matrix = cdist(X, self.current_population, metric='hamming')
        diversity_scores = np.mean(dist_matrix, axis=1)

        out["G"] = constraint_scores[:, None]
        out["F"] = -diversity_scores[:, None]


class ProblemMulti(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=2, n_constr=0)
        self.n_constraint = len(d4py.get_decl_model_constraints())
        self.n_events = len(initial_population[0])

    def _evaluate(self, X, out, *args, **kwargs):
        constraint_scores = self.evaluate_constraints_batch(X)
        dist_matrix = cdist(X, self.current_population, metric='hamming')
        diversity_scores = np.mean(dist_matrix, axis=1)

        # Weights
        diversity_weight = 0.2
        constraint_weight = 0.8

        # Normalization constants
        max_diversity = self.n_events
        max_constraint = self.n_constraint

        # Normalize
        normalized_diversity = diversity_scores / max_diversity
        normalized_constraints = constraint_scores / max_constraint

        # Weighted score computation (still arrays)
        weighted_diversity = diversity_weight * normalized_diversity
        weighted_constraint = constraint_weight * normalized_constraints


        out["F"] = [-weighted_diversity[:, None], weighted_constraint[:, None]]


class ProblemMultiObjectiveNovelty(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=3, n_constr=1)

        # Initialize a novelty archive (empty at start)
        self.novelty_archive = np.empty((0, trace_length))

    def _evaluate(self, X, out, *args, **kwargs):

        constraint_scores = self.evaluate_constraints_batch(X)
        dist_matrix = cdist(X, self.current_population, metric='hamming')
        diversity_scores = np.mean(dist_matrix, axis=1)


        if self.novelty_archive.shape[0] > 0:
            novelty_matrix = cdist(X, self.novelty_archive, metric='hamming')

        else:
            novelty_matrix = cdist(X, self.current_population, metric='hamming')

        # Use k-nearest neighbors to compute novelty
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

        # Optionally limit archive size
        archive_limit = 500
        if self.novelty_archive.shape[0] > archive_limit:
            self.novelty_archive = self.novelty_archive[-archive_limit:]



class ProblemSingleSing(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=1, n_constr=0)

    def _evaluate(self, X, out, *args, **kwargs):
        constraint_scores = self.evaluate_constraints_batch(X)
        dist_matrix = cdist(X, self.current_population, metric='hamming')
        diversity_scores = np.mean(dist_matrix, axis=1)

        # out["G"] = constraint_scores[:, None]
        out["F"] = -diversity_scores[:, None]



class ProblemSingleTrace(BaseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=1, n_constr=0)

    def _evaluate(self, X, out, *args, **kwargs):
        constraint_scores = self.evaluate_constraints_batch(X)
        # dist_matrix = cdist(X, self.current_population, metric='hamming')
        # diversity_scores = np.mean(dist_matrix, axis=1)

        # out["G"] = constraint_scores[:, None]
        out["F"] = constraint_scores[:, None]


