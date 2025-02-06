import numpy as np
from cvxopt.modeling import constraint
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from mutation import MyMutation
from crossover import TraceCrossover
from encoder import Encoder
from sampling import MySampling
from callback import UpdatePopulationCallback
from tools import Tools
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.D4PyEventLog import D4PyEventLog
import random
import pandas as pd
from problem import Problem_single_ElementWise, Problem_multi_ElementWise, Problem_single_ElementWise_noConstraints
from pymoo.algorithms.soo.nonconvex.ga import GA
import os
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from mutation import IntegerPolynomialMutation


class Setup:

    @staticmethod
    def setup_initial_population(trace_length, n_traces, activities_name, encoder):
        """
        Sets up the initial population and shared parameters for the given trace length.
        """

        # load CSV file with correct column names
        df = pd.read_csv("../declare_models/model1_initial_pop.csv", usecols=['Case ID', 'Activity'])

        # group activities by case, filter cases with at least 50 activities, and select the first 10 cases
        initial_population = [
                                 activities[:50] for activities in
                                 (df.groupby('Case ID')['Activity'].apply(list).values) if len(activities) >= 50
                             ][:10]

        initial_encoded_pop = [encoder.encode(trace) for trace in initial_population]
        lower_bounds = 0
        upper_bounds = len(activities_name) - 1

        return (
            initial_population,
            initial_encoded_pop,
            lower_bounds,
            upper_bounds,
            IntegerPolynomialMutation(prob=0.1, eta=20),
            SBX(prob=0.9, eta=15),
            MySampling(initial_population=initial_encoded_pop)
        )

    @staticmethod
    def initialize_shared_components(path_to_declareModel, trace_length):
        """
        Initializes shared components like the encoder, declare model, and event log.
        """

        declare = DeclareModel().parse_from_file(path_to_declareModel)
        activities_name = declare.get_model_activities()

        timestamps = Tools.generate_random_timestamps(trace_length)
        data = {
            'case:concept:name': ['1'] * trace_length,
            'concept:name': ['1'] * trace_length,
            'timestamp': pd.to_datetime(timestamps),
        }
        dataframe = pd.DataFrame(data)


        event_log = D4PyEventLog()
        event_log.log = dataframe
        event_log.timestamp_key = "timestamp"
        event_log.activity_key = "concept:name"
        event_log.to_eventlog()

        encoder = Encoder(activities_name)

        return encoder, declare, event_log, dataframe, activities_name

    @staticmethod
    def create_algorithm(algorithm_type, problem, pop_size, sampling, crossover, mutation):
        """
        Creates the appropriate algorithm (GA or NSGA2) based on the type.
        """
        algo_class = GA if algorithm_type == "single" else NSGA2
        return algo_class(
            problem=problem,
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            callback=UpdatePopulationCallback(problem=problem),
            eliminate_duplicates=False,
        )

    @staticmethod
    def create_problem(algorithm_type, trace_length, encoder, declare, initial_encoded_pop, lower_bounds, upper_bounds,
                       event_log, dataframe, constraints):
        """
        Creates the problem instance based on the algorithm type.
        """
        if algorithm_type == "single" and constraints == "no":
            problem_class = Problem_single_ElementWise_noConstraints
        else:
            problem_class = Problem_single_ElementWise if algorithm_type == "single" else Problem_multi_ElementWise

        return problem_class(
            trace_length=trace_length,
            encoder=encoder,
            d4py=declare,
            initial_population=initial_encoded_pop,
            xl=lower_bounds,
            xu=upper_bounds,
            event_log=event_log,
            dataframe=dataframe,
        )

    @staticmethod
    def log_results(f, ID, pop_size, trace_length, model, termination, algorithm_type, constraints,
                    exec_time=None, diversity_scores=None, constraint_scores=None, n_violations_scores=None, error=None):
        """
        Logs results to the file.
        """

        status = f"{exec_time:.2f}" if exec_time else "ERROR"

        diversity_score = abs(diversity_scores[-1]) / trace_length
        # determine diversity, constraint, n_violations
        diversity = f"{diversity_score:.2f}"
        constraint = f"{constraint_scores[-1]:.2f}" if constraint_scores else "NaN"
        n_violations = f"{n_violations_scores[-1]:.2f}" if n_violations_scores else "NaN"

        # write results to the CSV file
        f.write(
            f"{ID},{pop_size},{trace_length},{model},{termination},{algorithm_type},"
            f"{constraints},{status},{diversity},{constraint},{n_violations}\n"
        )

        if error:
            print(f"Error encountered with ID={ID}, Termination={termination}, Algorithm={algorithm_type}, Constraints={constraints}: {error}")
        else:
            print(f"Execution Time ({algorithm_type}): {exec_time:.2f} seconds")

    @staticmethod
    def plot_and_save_progress(ID, test_run, algorithm_type, constraints,  current_date,
                               diversity_scores=None, constraint_scores=None, n_violations_scores=None, n_generations=None):
        """
        Plot and save the fitness and constraint scores for the given algorithm (callback).
        """
        plt.figure(figsize=(15, 5))

        # diversity scores (always plot diversity scores)
        plt.subplot(1, 2 if constraints == "no" else (3 if algorithm_type == "multi" else 2), 1)
        plt.plot(range(n_generations), diversity_scores, label="Avg. Diversity", color="blue")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.title("Diversity Over Generations")
        plt.legend()
        plt.grid(True)

        # constraint scores (only if constraints == "yes")
        if constraints == "yes":
            plt.subplot(1, 3 if algorithm_type == "multi" else 2, 2)
            plt.plot(range(n_generations), constraint_scores, label="Avg. Constraint Score", color="orange")
            plt.xlabel("Generation")
            plt.ylabel("Constraint Score")
            plt.title("Constraint Scores Over Generations")
            plt.legend()
            plt.grid(True)

        # if multi-objective and constraints == "yes", plot n violation score
        if algorithm_type == "multi" and constraints == "yes":
            plt.subplot(1, 3, 3)
            plt.plot(range(n_generations), n_violations_scores, label="number violation score", color="green")
            plt.xlabel("Generation")
            plt.ylabel("Number Violation")
            plt.title("Number Violations Over Generations")
            plt.legend()
            plt.grid(True)

        # save the plot
        plot_name = f"ID_{ID}_run_{test_run}_{algorithm_type}_{constraints}_constraints.png"
        plt.tight_layout()
        os.makedirs(f"plots/run_{current_date}", exist_ok=True)
        plt.savefig(f"plots2/run_{current_date}/{plot_name}")
        plt.close()

    @staticmethod
    def save_feasible_traces(population, encoder, test_run, ID, algorithm_type, constraints, current_date):
        """
        Save the feasible traces from the final population into a file.
        """

        final_population = [individual.X.tolist() for individual in population]
        G = np.array([individual.G for individual in population])


        os.makedirs(f"results/encoded_traces_{current_date}", exist_ok=True)
        file_name = f"results/encoded_traces_{current_date}/ID_{ID}_run_{test_run}_{algorithm_type}_{constraints}_constraints.csv"

        # filter feasible solutions based on constraints
        if constraints == "yes":
            feasible_solutions = [trace for trace, g in zip(final_population, G) if g == 0]
        else:
            # if no constraints, consider the entire population as feasible
            feasible_solutions = final_population

        # decode the feasible solutions into readable traces
        decoded_traces = [encoder.decode(trace) for trace in feasible_solutions]

        # save the traces into the file
        with open(file_name, "w") as f:
            for trace in decoded_traces:
                # Convert the trace into event;event;event format
                encoded_trace = ";".join(map(str, trace))
                f.write(f"{encoded_trace}\n")

        # print(f"Feasible traces saved to {file_name}")


    def plot_progress(diversity_scores=None, constraint_scores=None, n_violations_scores=None, n_generations=None, constraints='yes', algorithm_type="single"):
        """
        Plot and save the fitness and constraint scores for the given algorithm (callback).
        """
        plt.figure(figsize=(15, 5))

        # diversity scores (always plot diversity scores)
        plt.subplot(1, 2 if constraints == "no" else (3 if algorithm_type == "multi" else 2), 1)
        plt.plot(range(n_generations), diversity_scores, label="Avg. Diversity", color="blue")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.title("Diversity Over Generations")
        plt.legend()
        plt.grid(True)

        # constraint scores (only if constraints == "yes")
        if constraints == "yes":
            plt.subplot(1, 3 if algorithm_type == "multi" else 2, 2)
            plt.plot(range(n_generations), constraint_scores, label="Avg. Constraint Score", color="orange")
            plt.xlabel("Generation")
            plt.ylabel("Constraint Score")
            plt.title("Constraint Scores Over Generations")
            plt.legend()
            plt.grid(True)

        # if multi-objective and constraints == "yes", plot n vio score
        if algorithm_type == "multi" and constraints == "yes":
            plt.subplot(1, 3, 3)
            plt.plot(range(n_generations), n_violations_scores, label="Number violation score", color="green")
            plt.xlabel("Generation")
            plt.ylabel("Number Violation")
            plt.title("Number Violations Over Generations")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()
