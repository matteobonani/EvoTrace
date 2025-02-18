from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from matplotlib import pyplot as plt


class Tools:

    @staticmethod
    def calculate_feature_range(population, variable_boundaries):
        """
        Calculate the feature range for each variable across the entire population.
        """

        population = np.array(population)  # Convert population to a NumPy array for faster operations
        num_variables = population.shape[1]  # Number of features/variables

        feature_ranges = []

        for j in range(num_variables):

            if variable_boundaries[j] == 1:  # non-dummy variable
                min_val = np.min(population[:, j])
                max_val = np.max(population[:, j])
                feature_ranges.append((min_val, max_val))
            else:  # dummy variable, range is always [0, 1]
                feature_ranges.append((0, 1))

        return feature_ranges

    @staticmethod
    def expand_population(initial_population, final_population_size, feature_range):
        """
        Expands the population by modifying the given traces until the target size is reached.
        """

        final_population = initial_population.copy()

        while len(final_population) < final_population_size:
            for trace in initial_population:

                new_trace = trace.copy()

                # TODO better to not modify more than once a dummy variable, and also check that inside the range, only one variable can be set as 1 [check mutations]
                # randomly select the number of columns to change
                num_columns_to_change = random.randint(1, len(feature_range))

                # selected the columns to modify in the trace
                columns_to_modify = random.sample(range(len(feature_range)), num_columns_to_change)

                for column_idx in columns_to_modify:

                    min_value, max_value = feature_range[column_idx]
                    new_value = random.randint(min_value, max_value)

                    new_trace[column_idx] = new_value


                final_population.append(new_trace)

        return final_population

    @staticmethod
    def generate_random_timestamps(length):
        start_date = datetime(2024, 12, 10, 10, 0, 0)
        timestamps = []

        for _ in range(length):
            random_minutes = random.randint(1, 10)
            start_date += timedelta(minutes=random_minutes)
            timestamps.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))

        return timestamps

    @staticmethod
    def calculate_diversity(trace, population):
        """
        Calculate diversity as the average Hamming Distance of the trace.
        """

        differences = np.sum(population != trace, axis=1)

        return np.mean(differences)

    @staticmethod
    def constraints_eval(dataframe, event_log, declare, traces):

        for x in traces:
            dataframe['concept:name'] = pd.DataFrame(x)
            event_log.log = dataframe
            event_log.to_eventlog()

            basic_checker = MPDeclareAnalyzer(log=event_log, declare_model=declare, consider_vacuity=False)
            conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
            metric_state = conf_check_res.get_metric(trace_id=0, metric="state")
            metric_num_violation = conf_check_res.get_metric(trace_id=0, metric="num_violations")

            print("Num_violations:")
            print(metric_num_violation)
            print("State:")
            print(metric_state)
            print("-------------------------------------------")

    @staticmethod
    def calculate_trace_diversity(encoder, n_trace, traces_path, n_events_per_trace):
        # read CSV assuming events are separated by semicolons
        df = pd.read_csv(traces_path, header=None, delimiter=";")

        # convert each row into a list of event sequences
        traces = df.values.tolist()
        encoded_traces = [encoder.encode(trace) for trace in traces]

        selected_trace = encoded_traces[n_trace]

        # compute diversity
        population = np.array(encoded_traces)
        trace_array = np.array(selected_trace)

        # compute Hamming distance
        diversity_score = Tools.calculate_diversity(trace_array, population)

        print(f"Selected Trace (Encoded): {selected_trace}")
        print(f"Diversity Score: {diversity_score / n_events_per_trace}")

    @staticmethod
    def calculate_overall_diversity(encoder, traces_path, n_events_per_trace):
        # read CSV assuming events are separated by semicolons
        df = pd.read_csv(traces_path, header=None, delimiter=";")

        # convert each row into a list of event sequences
        traces = df.values.tolist()
        encoded_traces = [encoder.encode(trace) for trace in traces]

        # convert encoded_traces to a numpy array for efficient processing
        population = np.array(encoded_traces)

        # Calculate the diversity for each trace
        diversity_scores = []
        for trace in population:
            diversity_score = Tools.calculate_diversity(trace, population)
            diversity_scores.append(diversity_score)

        # Overall diversity is the average of all individual trace diversities
        overall_diversity = np.mean(diversity_scores)

        print(f"Overall Diversity Score: {overall_diversity / n_events_per_trace}")

    @staticmethod
    def save_simple_solution(result_population, encoder):

        # assuming G[0] is the constraints value
        G = np.array([individual.G for individual in result_population])
        F = np.array([individual.F[1] for individual in result_population])

        final_population = [individual.X.tolist() for individual in result_population]
        feasible_solutions = [trace for trace, g in zip(final_population, F) if g == 0]


        # decode the feasible solutions into readable traces
        decoded_traces = [encoder.decode(trace) for trace in feasible_solutions]

        # save the traces into the file
        with open("simple_run_decoded_traces.csv", "w") as f:
            for trace in decoded_traces:
                # convert the trace into event;event;event format
                encoded_trace = ";".join(map(str, trace))
                f.write(f"{encoded_trace}\n")

    @staticmethod
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