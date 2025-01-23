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
from dataCollector import DataCollector
import os



class Setup:

    @staticmethod
    def setup_initial_population(trace_length, n_traces, activities_name, encoder):
        """
        Sets up the initial population and shared parameters for the given trace length.
        """

        initial_population = [[random.choice(activities_name) for _ in range(trace_length)] for _ in range(n_traces)]
        initial_encoded_pop = [encoder.encode(trace) for trace in initial_population]
        features_range = Tools.calculate_feature_range(initial_encoded_pop, [1] * trace_length)
        lower_bounds = [x[0] for x in features_range]
        upper_bounds = [x[1] for x in features_range]

        return (
            initial_population,
            initial_encoded_pop,
            features_range,
            lower_bounds,
            upper_bounds,
            MyMutation(feature_range=features_range),
            TraceCrossover(variable_boundaries=[1] * trace_length),
            MySampling(initial_population=initial_encoded_pop),
        )

    @staticmethod
    def initialize_shared_components(activities_name, path_to_declareModel):
        """
        Initializes shared components like the encoder, declare model, and event log.
        """

        data_collector = DataCollector()
        timestamps = Tools.generate_random_timestamps(len(activities_name))
        data = {
            'case:concept:name': ['1'] * len(activities_name),
            'concept:name': activities_name,
            'timestamp': pd.to_datetime(timestamps),
        }
        dataframe = pd.DataFrame(data)
        encoder = Encoder(activities_name)
        declare = DeclareModel().parse_from_file(path_to_declareModel)
        event_log = D4PyEventLog()
        event_log.log = dataframe
        event_log.timestamp_key = "timestamp"
        event_log.activity_key = "concept:name"

        return encoder, declare, event_log, dataframe, data_collector

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
    def log_results(f, ID, pop_size, trace_length, model, termination, algorithm_type, constraints, exec_time=None, error=None):
        """
        Logs results to the file.
        """
        status = f"{exec_time:.2f}" if exec_time else "ERROR"
        f.write(f"{ID},{pop_size},{trace_length},{model},{termination},{algorithm_type},{constraints},{status}\n")
        if error:
            print(f"Error encountered with ID={ID}, Termination={termination}, Algorithm={algorithm_type}, Constraints={constraints}: {error}")
        else:
            print(f"Execution Time ({algorithm_type}): {exec_time:.2f} seconds")

    @staticmethod
    def plot_and_save_progress(callback, ID, test_run, algorithm_type, constraints):
        """
        Plot and save the fitness and constraint scores for the given algorithm (callback).
        """
        data = callback.get_data()


        # diversity, constraint,number violation and number generation
        diversity_scores = data["diversity_history"]
        constraint_scores = data["constraint_history"]
        n_violations_scores = data.get("n_violations_history")
        number_generations = data["generations"]

        plt.figure(figsize=(15, 5))

        # diversity scores (always plot diversity scores)
        plt.subplot(1, 2 if constraints == "no" else (3 if algorithm_type == "multi" else 2), 1)
        plt.plot(range(number_generations), diversity_scores, label="Avg. Diversity", color="blue")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.title("Diversity Over Generations")
        plt.legend()
        plt.grid(True)

        # constraint scores (only if constraints == "yes")
        if constraints == "yes":
            plt.subplot(1, 3 if algorithm_type == "multi" else 2, 2)
            plt.plot(range(number_generations), constraint_scores, label="Avg. Constraint Score", color="orange")
            plt.xlabel("Generation")
            plt.ylabel("Constraint Score")
            plt.title("Constraint Scores Over Generations")
            plt.legend()
            plt.grid(True)

        # if multi-objective and constraints == "yes", plot diversity scores
        if algorithm_type == "multi" and constraints == "yes":
            plt.subplot(1, 3, 3)
            plt.plot(range(number_generations), n_violations_scores, label="Avg. Diversity Score", color="green")
            plt.xlabel("Generation")
            plt.ylabel("Number Violation")
            plt.title("Number Violations Over Generations")
            plt.legend()
            plt.grid(True)

        # save the plot
        plot_name = f"ID_{ID}_run_{test_run}_{algorithm_type}_{constraints}_constraints.png"
        plt.tight_layout()
        os.makedirs(f"plots/run_{test_run}", exist_ok=True)
        plt.savefig(f"plots/run_{test_run}/{plot_name}")
        plt.close()


