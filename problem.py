from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from pymoo.core.problem import ElementwiseProblem
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
import numpy as np
import pandas as pd

class Problem_multi_ElementWise(ElementwiseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl,xu, event_log, dataframe):

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

        decoded_trace = self.encoder.decode(trace)

        self.dataframe['concept:name'] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=False)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        metric_state = conf_check_res.get_metric(trace_id=0, metric="state")
        metric_num_violation = conf_check_res.get_metric(trace_id=0, metric="num_violations")

        metric_num_violation = np.array(metric_num_violation, dtype=np.float_)
        valid_values = metric_num_violation[np.isfinite(metric_num_violation)]
        violation_score = int(np.sum(valid_values))

        metric_state = np.array(metric_state)
        metric_state_inverted = 1 - metric_state  # inverted so 1 means violation and 0 means sat
        satisfy_score = np.mean(metric_state_inverted)

        return satisfy_score, violation_score

    def calculate_diversity(self, trace, population):
        """
        Calculate diversity as the average Hamming Distance of the trace.
        """

        population = np.array(population)
        trace_array = np.tile(trace, (population.shape[0], 1))

        # Calculate the Hamming distance (element-wise comparison)
        differences = np.sum(population != trace_array, axis=1)

        return np.mean(differences)


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


        out["G"] = [constraint_score[0]]  # feasible if <= 0

        out["F"] = [diversity_score, constraint_score[1]]


class Problem_single_ElementWise(ElementwiseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl,xu, event_log, dataframe):

        super().__init__(n_var=trace_length,
                         n_obj=1,  # objectives: diversity
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

        decoded_trace = self.encoder.decode(trace)

        self.dataframe['concept:name'] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=False)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        metric_state = conf_check_res.get_metric(trace_id=0, metric="state")
        metric_num_violation = conf_check_res.get_metric(trace_id=0, metric="num_violations")

        metric_num_violation = np.array(metric_num_violation, dtype=np.float_)
        valid_values = metric_num_violation[np.isfinite(metric_num_violation)]
        violation_score = int(np.sum(valid_values))

        metric_state = np.array(metric_state)
        metric_state_inverted = 1 - metric_state
        satisfy_score = np.mean(metric_state_inverted)


        return satisfy_score

    def calculate_diversity(self, trace, population):
        """
        Calculate diversity as the average Hamming Distance of the trace.
        """

        population = np.array(population)
        trace_array = np.tile(trace, (population.shape[0], 1))

        # Calculate the Hamming distance (element-wise comparison)
        differences = np.sum(population != trace_array, axis=1)

        return np.mean(differences)

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

        out["F"] = [diversity_score]

class Problem_single_ElementWise_noConstraints(ElementwiseProblem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl,xu, event_log, dataframe):

        super().__init__(n_var=trace_length,
                         n_obj=1,  # objectives: diversity
                         n_constr=0,
                         xl=xl,
                         xu=xu)

        self.trace_length = trace_length
        self.encoder = encoder
        self.d4py = d4py
        self.initial_population = initial_population
        self.current_population = self.initial_population
        self.event_log = event_log
        self.dataframe = dataframe


    def calculate_diversity(self, trace, population):
        """
        Calculate diversity as the average Hamming Distance of the trace.
        """

        population = np.array(population)
        trace_array = np.tile(trace, (population.shape[0], 1))

        # Calculate the Hamming distance (element-wise comparison)
        differences = np.sum(population != trace_array, axis=1)

        return np.mean(differences)

    def set_current_population(self, population):
        """
        Update the current population.
        """
        self.current_population = np.array(population)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a single trace.
        """

        # objective 1: Diversity
        diversity_score = -self.calculate_diversity(x, self.current_population) # negative because pymoo minimize, choose the lower value

        out["F"] = [diversity_score]


from pymoo.core.problem import Problem


class MyProblem_Problem(Problem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        """
        A problem with single objective and constraints, using pymoo's Problem class.
        """
        super().__init__(n_var=trace_length,      # number of variables (decision variables)
                         n_obj=2,                 # two objectives (constraint satisfaction and diversity)
                         n_constr=1,             # one constraint (violation score)
                         xl=xl,                  # lower bounds of decision variables
                         xu=xu)                  # upper bounds of decision variables

        # Assign other parameters to instance variables
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
        decoded_trace = self.encoder.decode(trace)

        self.dataframe['concept:name'] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=False)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        metric_state = conf_check_res.get_metric(trace_id=0, metric="state")
        metric_num_violation = conf_check_res.get_metric(trace_id=0, metric="num_violations")

        metric_num_violation = np.array(metric_num_violation, dtype=np.float_)
        valid_values = metric_num_violation[np.isfinite(metric_num_violation)]
        violation_score = int(np.sum(valid_values))

        metric_state = np.array(metric_state)
        metric_state_inverted = 1 - metric_state
        satisfy_score = np.mean(metric_state_inverted)

        return satisfy_score, violation_score

    def calculate_diversity(self, trace, population):
        """
        Calculate diversity as the average Hamming Distance of the trace.
        """
        population = np.array(population)
        trace_array = np.tile(trace, (population.shape[0], 1))

        # Calculate the Hamming distance (element-wise comparison)
        differences = np.sum(population != trace_array, axis=1)

        return np.mean(differences)

    def set_current_population(self, population):
        """
        Update the current population.
        """
        self.current_population = np.array(population)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the population X (batch processing).

        X: Population matrix, where each row represents an individual (solution).
        out: The output dictionary for objectives (F) and constraints (G).
        """

        # Evaluate the constraints and diversity for the entire population
        constraint_results = np.array([self.evaluate_constraints(individual) for individual in X])

        # Assuming evaluate_constraints returns a tuple of (constraint_score, violation_score)
        constraint_scores = constraint_results[:, 0]  # First column for constraint scores
        violation_scores = constraint_results[:, 1]  # Second column for violation scores

        # Calculate diversity scores for the entire population
        # Use broadcasting to calculate diversity for each individual
        diversity_scores = -np.array([self.calculate_diversity(individual, self.current_population) for individual in
                                      X])  # negative because pymoo minimizes

        # Store the objectives in out["F"] (minimize the first and second objectives)
        out["F"] = np.column_stack([violation_scores, diversity_scores])

        # Store the constraints in out["G"] (feasible solutions have G <= 0)
        out["G"] = np.array(constraint_scores)  # feasible if G <= 0 (constraint satisfaction)

class MyProblem_Problem2(Problem):
    def __init__(self, trace_length, encoder, d4py, initial_population, xl, xu, event_log, dataframe):
        """
        A problem with single objective and constraints, using pymoo's Problem class.
        """
        super().__init__(n_var=trace_length,      # number of variables (decision variables)
                         n_obj=1,                 # two objectives (constraint satisfaction and diversity)
                         n_constr=1,             # one constraint (violation score)
                         xl=xl,                  # lower bounds of decision variables
                         xu=xu)                  # upper bounds of decision variables

        # Assign other parameters to instance variables
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
        decoded_trace = self.encoder.decode(trace)

        self.dataframe['concept:name'] = pd.DataFrame(decoded_trace)
        self.event_log.log = self.dataframe
        self.event_log.to_eventlog()

        basic_checker = MPDeclareAnalyzer(log=self.event_log, declare_model=self.d4py, consider_vacuity=False)
        conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
        metric_state = conf_check_res.get_metric(trace_id=0, metric="state")
        metric_num_violation = conf_check_res.get_metric(trace_id=0, metric="num_violations")

        metric_num_violation = np.array(metric_num_violation, dtype=np.float_)
        valid_values = metric_num_violation[np.isfinite(metric_num_violation)]
        violation_score = int(np.sum(valid_values))

        metric_state = np.array(metric_state)
        metric_state_inverted = 1 - metric_state
        satisfy_score = np.mean(metric_state_inverted)

        return satisfy_score, violation_score

    def calculate_diversity(self, trace, population):
        """
        Calculate diversity as the average Hamming Distance of the trace.
        """
        population = np.array(population)
        trace_array = np.tile(trace, (population.shape[0], 1))

        # Calculate the Hamming distance (element-wise comparison)
        differences = np.sum(population != trace_array, axis=1)

        return np.mean(differences)

    def set_current_population(self, population):
        """
        Update the current population.
        """
        self.current_population = np.array(population)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the population X (batch processing).

        X: Population matrix, where each row represents an individual (solution).
        out: The output dictionary for objectives (F) and constraints (G).
        """

        # Evaluate the constraints and diversity for the entire population
        constraint_results = np.array([self.evaluate_constraints(individual) for individual in X])

        # Assuming evaluate_constraints returns a tuple of (constraint_score, violation_score)
        constraint_scores = constraint_results[:, 0]  # First column for constraint scores

        # Calculate diversity scores for the entire population
        # Use broadcasting to calculate diversity for each individual
        diversity_scores = -np.array([self.calculate_diversity(individual, self.current_population) for individual in
                                      X])  # negative because pymoo minimizes

        # Store the objectives in out["F"] (minimize the first and second objectives)
        out["F"] = np.column_stack(diversity_scores)

        # Store the constraints in out["G"] (feasible solutions have G <= 0)
        out["G"] = np.array(constraint_scores)  # feasible if G <= 0 (constraint satisfaction)

