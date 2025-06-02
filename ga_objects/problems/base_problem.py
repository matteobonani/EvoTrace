from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from pymoo.core.problem import Problem
from joblib import Parallel, delayed
from pywin.mfc.object import Object
from Declare4Py.ProcessModels import DeclareModel
from Declare4Py.D4PyEventLog import D4PyEventLog

class BaseElementWiseProblem(ElementwiseProblem):
    """
    Base class for defining element-wise optimization problems.

    This class sets up common components for problems that involve optimizing
    sequences (traces) by evaluating diversity and constraint satisfaction.

    Parameters:
    ----------
    trace_length : int
        The length of a trace.
    encoder : Encoder
        The encoder to convert between activity names and numerical representations.
    d4py : DeclareModel
        The declarative model used for evaluating constraint satisfaction.
    initial_population : list
        The initial population of candidate traces.
    xl : np.ndarray or int
        The lower bounds for decision variables.
    xu : np.ndarray or int
        The upper bounds for decision variables.
    event_log : D4PyEventLog
        The event log containing process execution traces.
    dataframe : pd.DataFrame
        The dataframe representation of the event log.
    n_obj : int
        The number of objectives in the optimization problem.
    n_constr : int
        The number of constraints in the optimization problem.
    """

    def __init__(self, trace_length: int,
                 encoder: Object,
                 d4py: DeclareModel,
                 initial_population: list,
                 xl: np.ndarray | int,
                 xu: np.ndarray | int,
                 event_log: D4PyEventLog,
                 dataframe: Object,
                 n_obj: int,
                 n_constr: int):
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

class BaseProblem(Problem):
    """
    Base class for defining trace-based optimization problems.

    This class sets up the core components required for optimization problems
    involving sequences (traces), providing integration with encoding schemes,
    process mining utilities, and event log data for evaluation.

    Parameters:
    ----------
    trace_length : int
        The length of a trace.
    encoder : Encoder
        The encoder to convert between activity names and numerical representations.
    d4py : DeclareModel
        The declarative model used for evaluating constraint satisfaction.
    initial_population : list
        The initial population of candidate traces.
    xl : np.ndarray or int
        The lower bounds for decision variables.
    xu : np.ndarray or int
        The upper bounds for decision variables.
    event_log : D4PyEventLog
        The event log containing process execution traces.
    dataframe : pd.DataFrame
        The dataframe representation of the event log.
    n_obj : int
        The number of objectives in the optimization problem.
    n_constr : int
        The number of constraints in the optimization problem.
    """
    def __init__(self,
                 trace_length: int,
                 encoder: Object,
                 d4py: DeclareModel,
                 initial_population: list,
                 xl: np.ndarray | int,
                 xu: np.ndarray | int,
                 event_log: D4PyEventLog,
                 dataframe: Object,
                 n_obj: int,
                 n_constr: int):
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