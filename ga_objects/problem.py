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



class BaseProblem(ElementwiseProblem):
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


class ProblemMultiElementWise(BaseProblem):
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


class ProblemMultiNoConstElementWise(BaseProblem):
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


class ProblemSingleElementWise(BaseProblem):
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


class ProblemSingleElementWiseNoConstraints(BaseProblem):
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


class ProblemSingle(Problem):
    """
    Single-objective optimization problem.

    Objective:
    - Maximize diversity (minimize similarity to existing population).

    Constraints:
    - Constraint satisfaction score must be <= 0 for feasibility.

    Parameters:
    ----------
    trace_length : int
        The length of each trace.
    encoder : Encoder
        Converts between activity names and integer representations.
    d4py : DeclareForPyModel
        The declarative model used for constraint checking.
    initial_population : list
        The initial population of traces.
    xl : np.ndarray or int
        The lower bound of decision variables.
    xu : np.ndarray or int
        The upper bound of decision variables.
    event_log : EventLog
        The event log containing process execution traces.
    dataframe : pd.DataFrame
        The dataframe representation of the event log.
    """

    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(n_var=trace_length, n_obj=1, n_constr=1, xl=xl, xu=xu)

        self.trace_length = trace_length
        self.encoder = encoder
        self.d4py = d4py
        self.initial_population = np.array(initial_population)
        self.current_population = self.initial_population
        self.event_log = event_log
        self.dataframe = dataframe


    def evaluate_constraints_batch(self, X):
        """
        Evaluate constraints for a batch of traces in parallel.

        Parameters:
        ----------
        X : np.ndarray
            A batch of traces.

        Returns:
        -------
        np.ndarray
            An array of constraint scores for each trace.
        """

        constraint_scores = Parallel(n_jobs=-1)(delayed(self.evaluate_constraints)(trace) for trace in X)
        return np.array(constraint_scores)

    def evaluate_constraints(self, trace):
        """
        Evaluate constraint violations for a single trace.

        Parameters:
        ----------
        trace : np.ndarray
            A single encoded process trace.

        Returns:
        -------
        int
            The total number of violated constraints.
        """
        decoded_trace = self.encoder.decode(trace)
        self.dataframe["concept:name"] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog() # 1.3s | pop = 3000 - trace_length = 50 - model1

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=True)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run() # 1.7s | pop = 3000 - trace_length = 50 - model1
        metric_state = np.array(conf_check_res.get_metric(trace_id=0, metric="state"))
        metric_state_inverted = 1 - metric_state

        return np.sum(metric_state_inverted)

    def evaluate_all_constraints(self, population):

        start_time = time.perf_counter()
        decoded_pop = self.encoder.decode(population)

        flattened_decoded_pop = np.array(decoded_pop).flatten()
        end_time = time.perf_counter()
        print(f"Time elapsed in decoding and flatten: {end_time - start_time:.6f} seconds")

        self.dataframe["concept:name"] = pd.DataFrame(flattened_decoded_pop)



        self.event_log.log = self.dataframe


        start_time = time.perf_counter()
        self.event_log.to_eventlog()
        end_time = time.perf_counter()
        print(f"Time elapsed eventlog.to_eventlog: {end_time - start_time:.6f} seconds")


        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=True)


        start_time = time.perf_counter()
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        end_time = time.perf_counter()
        print(f"Time elapsed basic_checker.run(): {end_time - start_time:.6f} seconds")

        metric_states = np.array(conf_check_res.get_metric(metric="state"))


        metric_states_inverted = 1 - metric_states


        return np.sum(metric_states_inverted, axis=1)

    def set_current_population(self, population):
        """Update the current population."""
        self.current_population = np.array(population)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate multiple traces one-by-one in a loop.

        Parameters:
        ----------
        X : np.ndarray
            A batch of traces.
        out : dict
            Dictionary to store results.
        """

        constraint_scores = self.evaluate_constraints_batch(X)
        # constraint_scores = self.evaluate_all_constraints(X)

        # pairwise Hamming distance matrix
        dist_matrix = cdist(X,self.current_population, metric='hamming')

        # n = dist_matrix.shape[0]
        # mean_per_trace = (np.sum(dist_matrix, axis=1) - np.diag(dist_matrix)) / (n - 1)

        diversity_scores = np.mean(dist_matrix, axis=1)

        out["G"] = constraint_scores[:, None]
        out["F"] = -diversity_scores[:, None]




