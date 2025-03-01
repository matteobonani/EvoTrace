from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from matplotlib import pyplot as plt


class Tools:
    """A utility class that provides various helper methods."""

    @staticmethod
    def generate_random_timestamps(length):
        """
        Generates a list of random timestamps, starting from a fixed date (2024-12-10 10:00:00).
        Each subsequent timestamp is incremented by a random number of minutes (between 1 and 10).

        Parameters:
        ----------
        length : int
            The number of timestamps to generate.

        Returns:
        -------
        list
            A list of timestamps in the format 'YYYY-MM-DD HH:MM:SS'.
        """
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
        Computes the diversity of a given trace by calculating its average Hamming Distance
        compared to the rest of the population.

        Parameters:
        ----------
        trace : list or numpy array
            A single trace (encoded as a sequence).

        population : numpy array
            The entire population of traces.

        Returns:
        -------
        float
            The average Hamming Distance of the trace relative to the population.
        """

        differences = np.sum(population != trace, axis=1)

        return np.mean(differences)

    @staticmethod
    def constraints_eval(dataframe, event_log, declare, traces):
        """
        Evaluates a set of traces against predefined Declare constraints and logs the results.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            DataFrame used for storing trace data.

        event_log :  Event_log object
            An event log object that stores the traces.

        declare : Declare object
            A Declare model containing constraints.

        traces : list
            A list of traces to be evaluated.
        """
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
        """
        Calculates the diversity of a selected trace compared to the population and logs the results.

        Parameters:
        ----------
        encoder : Encoder object
            Encoder used to encode traces into numerical format.

        n_trace : int
            Index of the selected trace.

        traces_path : str
            Path to the CSV file containing traces.

        n_events_per_trace : int
            Number of events per trace.
        """
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
        """
        Computes the overall diversity of a population based on Hamming distance and logs the results.

        Parameters:
        ----------
        encoder : Encoder object
            Encoder used to encode traces into numerical format.

        traces_path : str
            Path to the CSV file containing traces.

        n_events_per_trace : int
            Number of events per trace.
        """
        # read CSV assuming events are separated by semicolons
        df = pd.read_csv(traces_path, header=None, delimiter=";")

        # convert each row into a list of event sequences
        traces = df.values.tolist()
        encoded_traces = [encoder.encode(trace) for trace in traces]

        # convert encoded_traces to a numpy array for efficient processing
        population = np.array(encoded_traces)

        # calculate the diversity for each trace
        diversity_scores = []
        for trace in population:
            diversity_score = Tools.calculate_diversity(trace, population)
            diversity_scores.append(diversity_score)

        overall_diversity = np.mean(diversity_scores)

        print(f"Overall Diversity Score: {overall_diversity / n_events_per_trace}")

    @staticmethod
    def save_simple_solution(result_population, encoder, exec_time, algorithm, population, trace_length, constraint_location, constraint_index=None):
        """
        Saves feasible solutions from the result population to a CSV file, considering constraints
        based on either the 'G' or 'F' attributes.

        Parameters:
        ----------
        result_population : list
            The final population after the optimization process.

        encoder : Encoder object
            Encoder used for decoding numerical representations into readable traces.

        exec_time : float
            The execution time of the algorithm in seconds.

        algorithm : str
            Name of the algorithm used.

        population : int
            The total population size.

        trace_length : int
            Length of each trace in the population.

        constraint_location : str
            Indicates whether constraints are stored in 'G' (general constraints) or 'F' (specific fitness constraints).

        constraint_index : int, optional
            The index of the constraint within 'F', required if constraint_location is 'F'.
        """

        if constraint_location == "G":
            constraints_array = np.array([individual.G for individual in population])
        else:  # constraint_location == "F"
            constraints_array = np.array([individual.F[constraint_index] for individual in population])

        final_population = [individual.X.tolist() for individual in result_population]
        feasible_solutions = [trace for trace, g in zip(final_population, constraints_array) if g == 0]


        # decode the feasible solutions into readable traces
        decoded_traces = [encoder.decode(trace) for trace in feasible_solutions]

        # save the traces into the file
        with open("simple_run_decoded_traces.csv", "w") as f:
            for trace in decoded_traces:
                # convert the trace into event;event;event format
                encoded_trace = ";".join(map(str, trace))
                f.write(f"{encoded_trace}\n")

            f.write(f"Execution Time: {exec_time:.2f};Algorithm: {algorithm};Population: {population};Trace Length: {trace_length}\n")

    @staticmethod
    def plot_progress(n_subplots, diversity_scores=None, constraint_scores=None,
                      n_violations_scores=None, n_generations=None):
        """
        Plots the evolution of diversity, constraint scores, and number of violations
        across generations in an optimization process.

        Parameters:
        ----------
        n_subplots : int
            The number of subplots to create, based on the available data.

        diversity_scores : list, optional
            List of diversity scores across generations.

        constraint_scores : list, optional
            List of constraint satisfaction scores across generations.

        n_violations_scores : list, optional
            List of the number of violations across generations.

        n_generations : int
            The total number of generations.
        """
        plt.figure(figsize=(5 * n_subplots, 5))

        subplot_idx = 1

        # plot diversity scores
        plt.subplot(1, n_subplots, subplot_idx)
        plt.plot(range(n_generations), diversity_scores, label="Avg. Diversity", color="blue")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.title("Diversity Over Generations")
        plt.legend()
        plt.grid(True)
        subplot_idx += 1  # Move to next subplot

        # plot constraint scores if provided
        if constraint_scores is not None and subplot_idx <= n_subplots:
            plt.subplot(1, n_subplots, subplot_idx)
            plt.plot(range(n_generations), constraint_scores, label="Avg. Constraint Score", color="orange")
            plt.xlabel("Generation")
            plt.ylabel("Constraint Score")
            plt.title("Constraint Scores Over Generations")
            plt.legend()
            plt.grid(True)
            subplot_idx += 1

        # plot number of violations if provided
        if n_violations_scores is not None and subplot_idx <= n_subplots:
            plt.subplot(1, n_subplots, subplot_idx)
            plt.plot(range(n_generations), n_violations_scores, label="Number of Violations", color="green")
            plt.xlabel("Generation")
            plt.ylabel("Number of Violations")
            plt.title("Number of Violations Over Generations")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

