from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer

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
        diversity_score = -self.calculate_diversity(x, self.current_population)

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
        diversity_score = -self.calculate_diversity(x, self.current_population)

        diversity_weight = 0.4
        constraint_weight = 0.6

        # normalize the diversity to range between 0 and 1
        min_diversity = -50  # lower bound for diversity
        max_diversity = 0  # upper bound
        normalized_diversity = (diversity_score - min_diversity) / (max_diversity - min_diversity)

        # normalize the constraint evaluation to range between 0 and 1
        max_constraint = 10  # upper bound
        normalized_constraint = min(constraint_score / max_constraint, 1)

        # combine the weighted objectives
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
        diversity_score = -self.calculate_diversity(x, self.current_population)
        out["G"] = [constraint_score]  # feasible if <= 0
        out["F"] = [diversity_score]


class ProblemSingleElementWiseNoConstraints(BaseProblem):
    """
    Single-objective problem without constraints.

    Objective:
    - Maximize diversity (minimize similarity to existing population)
    """
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        super().__init__(trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe, n_obj=1, n_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        diversity_score = -self.calculate_diversity(x, self.current_population)
        out["F"] = [diversity_score]
