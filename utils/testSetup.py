import random
from matplotlib import pyplot as plt
from utils.encoder import Encoder
from ga_objects.sampling import MySampling
from ga_objects.callback import UpdatePopulationCallback
from utils.tools import Tools
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
import os
from pymoo.operators.crossover.sbx import SBX
from ga_objects.mutation import IntegerPolynomialMutation
from ga_objects.problem import *
from pymoo.optimize import minimize


class Setup:
    """Utility class for managing setup, execution, and result tracking of optimization experiments."""

    # -----------------------
    # Data Preparation
    # -----------------------

    @staticmethod
    def initial_random_population(path_to_declareModel, trace_length, trace_number):
        declare = DeclareModel().parse_from_file(path_to_declareModel)
        activities = declare.get_model_activities()

        return [[random.choice(activities) for _ in range(trace_length)] for _ in range(trace_number)]

    @staticmethod
    def extract_traces(path, trace_length=50):
        activities = pd.read_csv(path, usecols=['concept:name'])['concept:name'].tolist()

        return [activities[i:i + trace_length] for i in range(0, len(activities), trace_length)]

    @staticmethod
    def initialize_shared_components(path_to_declareModel, trace_length, initial_population):
        declare = DeclareModel().parse_from_file(path_to_declareModel)
        activities = declare.get_model_activities()
        encoder = Encoder(activities)
        encoded_pop = encoder.encode(initial_population)

        timestamps = Tools.generate_random_timestamps(trace_length)
        df = pd.DataFrame({
            'case:concept:name': ['1'] * trace_length,
            'concept:name': ['1'] * trace_length,
            'timestamp': pd.to_datetime(timestamps),
        })

        event_log = D4PyEventLog()
        event_log.log = df
        event_log.timestamp_key = "timestamp"
        event_log.activity_key = "concept:name"
        event_log.to_eventlog()

        return encoder, declare, event_log, df, encoded_pop, 0, len(activities) - 1

    # -----------------------
    # Optimization Components
    # -----------------------

    @staticmethod
    def create_problem(problem_class, trace_length, encoder, declare, encoded_pop, xl, xu, event_log, dataframe):
        return problem_class(
            trace_length=trace_length,
            encoder=encoder,
            d4py=declare,
            initial_population=encoded_pop,
            xl=xl,
            xu=xu,
            event_log=event_log,
            dataframe=dataframe,
        )

    @staticmethod
    def create_algorithm(algo_class, problem, pop_size, sampling, crossover, mutation):
        return algo_class(
            problem=problem,
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            callback=UpdatePopulationCallback(),
            eliminate_duplicates=False,
        )

    @staticmethod
    def run_result(problem, algorithm, termination, verbose):
        result = minimize(problem, algorithm, termination=termination, verbose=verbose)
        data = result.algorithm.callback.get_data()
        n_generations = result.algorithm.n_gen
        execution_time = result.exec_time
        population = result.pop

        return (
            data,
            population,
            n_generations,
            execution_time
        )

    @staticmethod
    def invert_weighted_normalization(div_scores, div_weight, cons_scores, cons_weight, max_constraint, max_diversity):

        if div_weight == 1 and cons_weight == 1:
            div_scores = [abs(score) for score in div_scores]
        else:
            div_scores = [abs(score) / div_weight * max_diversity for score in div_scores]
            cons_scores = [score / cons_weight * max_constraint for score in cons_scores]

        return div_scores, cons_scores

    # -----------------------
    # Results and Plotting
    # -----------------------

    @staticmethod
    def plot_evolution(ID, run, problem_type, n_subplots, diversity, n_gen, constraint=None, violations=None, save_path=""):
        plt.figure(figsize=(5 * n_subplots, 5))
        idx = 1
        n_gen = n_gen - 1

        plt.subplot(1, n_subplots, idx)
        plt.plot(range(n_gen), diversity, label="Avg. Diversity", color="blue")
        plt.title("Diversity Over Generations")
        plt.xlabel("Generation"); plt.ylabel("% Diversity"); plt.grid(); plt.legend()
        idx += 1

        if constraint and idx <= n_subplots:
            plt.subplot(1, n_subplots, idx)
            plt.plot(range(n_gen), constraint, label="Constraint Score", color="orange")
            plt.title("Constraint Scores"); plt.xlabel("Generation"); plt.ylabel("Score"); plt.grid(); plt.legend()
            idx += 1

        if violations and idx <= n_subplots:
            plt.subplot(1, n_subplots, idx)
            plt.plot(range(n_gen), violations, label="Violations", color="green")
            plt.title("Violations"); plt.xlabel("Generation"); plt.ylabel("# Violations"); plt.grid(); plt.legend()

        os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"ID_{ID}_run_{run}_{problem_type}.png"))
        plt.close()

    @staticmethod
    def record_experiment_results(file, run, ID, pop_size, trace_length, model, problem_type, mutation_type, termination, n_gen, exec_time, diversity=None, constraint=None, violations=None, error=None):

        status = f"{exec_time:.2f}" if exec_time else "ERROR"
        diversity_score = abs(diversity[-1]) if diversity else "NaN"
        constraint_score = f"{constraint[-1]:.2f}" if constraint else "NaN"
        violations_score = f"{violations[-1]:.2f}" if violations else "NaN"

        file.write(f"{ID},{pop_size},{trace_length},{model},{problem_type.__name__},{type(mutation_type).__name__},{type(termination).__name__},{status},{diversity_score:.2f},{constraint_score},{n_gen},{run}\n")

        if error:
            print(f"Error with ID={ID}, Problem={problem_type.__name__}, Mutation={type(mutation_type).__name__}: {error}")
        else:
            print(f"Execution Time ({problem_type.__name__}): {exec_time:.2f}s")

    @staticmethod
    def save_valid_solutions(population, encoder, run, ID, problem_type, save_path, constraint_location, constraint_index=None):
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"ID_{ID}_run_{run}_{problem_type}.csv")

        population_list = [ind.X.tolist() for ind in population]
        constraints = np.array([ind.G if constraint_location == "G" else ind.F[constraint_index] for ind in population])
        feasible = [trace for trace, c in zip(population_list, constraints) if c <= 0]

        decoded = encoder.decode(feasible)
        with open(file_path, "w") as f:
            for trace in decoded:
                f.write(";".join(map(str, trace)) + "\n")
