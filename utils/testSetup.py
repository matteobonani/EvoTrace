import random
from matplotlib import pyplot as plt
from utils.encoder import Encoder
from ga_objects.sampling import MySampling
from ga_objects.callback import UpdatePopulationCallback
from utils.tools import Tools
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.D4PyEventLog import D4PyEventLog
import os
from pymoo.operators.crossover.sbx import SBX
from ga_objects.mutation import IntegerPolynomialMutation
from ga_objects.problem import *
from pymoo.optimize import minimize
import pm4py


class Setup:
    """A utility class for managing the setup and execution of optimization experiments."""

    @staticmethod
    def invert_wighted_normalization(diversity_scores, weighted_diversity_value, constraint_scores, weighted_constraint_value, trace_length):


        if weighted_diversity_value == 1 and weighted_constraint_value == 1:
            diversity_scores = [abs(score) for score in diversity_scores]

        else:
            diversity_scores = [abs(score) / weighted_diversity_value for score in diversity_scores]
            constraint_scores = [score / weighted_constraint_value * 10 for score in constraint_scores]


        return diversity_scores, constraint_scores



    @staticmethod
    def run_result(problem, algorithm, termination, verbose):
        """
        Executes the optimization process and returns relevant experiment data.

        Parameters:
        ----------
        problem : Problem object
            The problem instance that defines the optimization task.

        algorithm : Algorithm object
            The algorithm used for optimization.

        termination : Termination object
            The termination used for the optimization.

        verbose : bool
            Whether to display progress information during execution.

        Returns:
        -------
        exec_time : float
            The execution time of the optimization process.

        constraint_scores : list
            A list of constraint scores over the course of the optimization.

        first_objective : list
            A list of values for the first objective over the optimization.

        second_objective : list
            A list of values for the second objective over the optimization.

        n_generations : int
            The number of generations run during the optimization.

        population : list
            The final population after optimization.
        """
        result = minimize(problem, algorithm, termination=termination, verbose=verbose)
        exec_time = result.exec_time

        # extract callback data
        data = result.algorithm.callback.get_data()
        first_objective = data.get("first_objective", None)
        constraint_scores = data.get("constraint_history", None)
        second_objective = data.get("second_objective", None)
        n_generations = data.get("generations", None)
        population = result.pop

        return exec_time, constraint_scores, first_objective, second_objective, n_generations, population



    @staticmethod
    def initial_population(path_to_population, trace_length, trace_number):
        """
        Loads and prepares the initial population based on a CSV file.

        Parameters:
        ----------
        path_to_population : str
            Path to the CSV file containing the initial population data.

        trace_length : int
            Length of the traces in the file.

        trace_number : int
            Number of traces to be included in the initial population.

        Returns:
        -------
        initial_population : list
            A list of the initial population traces.
        """
        # load CSV file with correct column names
        df = pd.read_csv(path_to_population, usecols=['Case ID', 'Activity'])

        # group activities by case, filter cases with at least 50 activities, and select the first 10 cases
        initial_population = [
                                 activities[:trace_length] for activities in
                                 (df.groupby('Case ID')['Activity'].apply(list).values) if len(activities) >= trace_length
                             ][:trace_number]

        return initial_population

    @staticmethod
    def extract_traces(path_to_population, trace_length=50):
        """
        Extract traces from a CSV file, assuming each trace has exactly `trace_length` activities.

        Parameters:
        - path_to_population (str): Path to the CSV file.
        - trace_length (int): Number of activities per trace (default is 50).

        Returns:
        - list: A list of traces, where each trace is a list of `trace_length` activities.
        """

        # Load CSV and read only the 'concept:name' column
        df = pd.read_csv(path_to_population, usecols=['concept:name'])

        # Convert column to a list of activities
        activities = df['concept:name'].tolist()

        # Split activities into chunks of `trace_length`
        traces = [activities[i:i + trace_length] for i in range(0, len(activities), trace_length)]

        return traces
    @staticmethod
    def initial_random_population(path_to_declareModel, trace_length, trace_number):
        """
        Generates n random traces, each of a specified length.
        """

        declare = DeclareModel().parse_from_file(path_to_declareModel)
        activities_name = declare.get_model_activities()
        traces = [[random.choice(activities_name) for _ in range(trace_length)] for _ in range(trace_number)]
        return traces




    @staticmethod
    def setup_initial_population(activities_name, encoder):
        """
        Sets up the initial population and shared parameters for the given trace length.

        Parameters:
        ----------
        activities_name : list
            List of activities in the Declare model.

        encoder : Encoder object
            Encoder used to encode and decode population traces.

        Returns:
        -------
        initial_population : list
            A list of the initial population traces.

        initial_encoded_pop : list
            A list of encoded initial population traces.

        lower_bounds : int
            Lower bounds of the search space.

        upper_bounds : int
            Upper bounds of the search space.

        mutation : Mutation object
            Mutation used in the genetic algorithm.

        crossover : Crossover object
            Crossover used in the genetic algorithm.

        sampling : Sampling object
            Sampling used to generate the population samples.
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
    def initialize_shared_components(path_to_declareModel, trace_length, initial_population, pop_size):
        """
        Initializes shared components like the encoder, Declare model, and event log.

        Parameters:
        ----------
        path_to_declareModel : str
            Path to the Declare model.

        trace_length : int
            The length of the traces.

        initial_population : list or array
            The initial population used in the optimization.

        Returns:
        -------
        encoder : Encoder object
            The encoder instance used to encode traces.

        declare : DeclareModel
            The parsed Declare model.

        event_log : D4PyEventLog
            The event log with the appropriate data.

        dataframe : pd.DataFrame
            A DataFrame containing the event log data.

        initial_encoded_pop : list
            A list of encoded initial population traces.

        lower_bounds : int
            Lower bounds of the search space.

        upper_bounds : int
            Upper bounds of the search space.
        """

        declare = DeclareModel().parse_from_file(path_to_declareModel)
        activities_name = declare.get_model_activities()

        timestamps = Tools.generate_random_timestamps(trace_length)
        data = {
            'case:concept:name': ['1'] * trace_length,
            'concept:name': ['1'] * trace_length,
            'timestamp': pd.to_datetime(timestamps),
        }

        # case_concept_name = np.array([[str(i + 1)] * trace_length for i in range(pop_size)])
        # concept_name = np.full((pop_size, trace_length), '1')  # 2D array filled with '1'
        # timestamps = pd.to_datetime(timestamps)
        # repeated_timestamps = np.tile(timestamps, (pop_size, 1))
        #
        #
        # data = {
        #     'case:concept:name': case_concept_name.flatten(),  # convert 2D array to list of lists
        #     'concept:name': concept_name.flatten(),  # convert 2D array to list of lists
        #     'timestamp': repeated_timestamps.flatten()  # convert 2D array to list of lists
        # }


        dataframe = pd.DataFrame(data)

        event_log = D4PyEventLog()
        event_log.log = dataframe
        event_log.timestamp_key = "timestamp"
        event_log.activity_key = "concept:name"
        event_log.to_eventlog()

        encoder = Encoder(activities_name)

        # initial_encoded_pop = [encoder.encode(trace) for trace in initial_population]
        initial_encoded_pop = encoder.encode(initial_population)
        lower_bounds = 0
        upper_bounds = len(activities_name) - 1

        return encoder, declare, event_log, dataframe, initial_encoded_pop, lower_bounds, upper_bounds

    @staticmethod
    def create_algorithm(algo_class, problem, pop_size, sampling, crossover, mutation):
        """
        Creates an optimization algorithm with the given configuration.

        Parameters:
        ----------
        algo_class : Algorithm object
            The algorithm class to be used.

        problem : Problem object
            The optimization problem instance.

        pop_size : int
            The size of the population in the algorithm.

        sampling : Sampling object
            Sampling method used in the algorithm.

        crossover : Crossover object
            Crossover operator used in the algorithm.

        mutation : Mutation object
            Mutation operator used in the algorithm.

        Returns:
        -------
        algorithm : Algorithm object
            The instantiated algorithm ready for execution.
        """
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
    def create_problem(problem_class, trace_length, encoder, declare, initial_encoded_pop, lower_bounds, upper_bounds,
                       event_log, dataframe):
        """
        Creates an optimization problem with the given configuration.

        Parameters:
        ----------
        problem_class : Problem
            The problem class defining the optimization problem.

        trace_length : int
            The length of the traces in the problem.

        encoder : Encoder object
            The encoder instance for encoding the traces.

        declare : DeclareModel
            The Declare model used for constraint checking.

        initial_encoded_pop : list
            The initial population (encoded).

        lower_bounds : int
            Lower bounds of the search space.

        upper_bounds : int
            Upper bounds of the search space.

        event_log : D4PyEventLog
            The event log used in the problem.

        dataframe : pd.DataFrame
            DataFrame containing the event log data.

        Returns:
        -------
        problem : Problem object
            The instantiated problem for optimization.
        """
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
    def plot_evolution(ID, test_run, problem_type, n_subplots, diversity_scores,
                               constraint_scores=None, n_violations_scores=None,
                               n_generations=None, save_path=""):
        """
        Plot and save the fitness and constraint scores over generations.

        Parameters:
        ----------
        ID : str
            Experiment identifier.

        test_run : int
            The run number.

        problem_type : str
            Type of the optimization problem.

        n_subplots : int
            Number of subplots to display in the figure.

        diversity_scores : list
            List of diversity scores over generations.

        constraint_scores : list, optional
            List of constraint scores over generations.

        n_violations_scores : list, optional
            List of violation scores over generations.

        n_generations : int
            Number of generations in the optimization.

        save_path : str
            Path where the plot will be saved.
        """

        plt.figure(figsize=(5 * n_subplots, 5))

        subplot_idx = 1

        # plot diversity scores
        plt.subplot(1, n_subplots, subplot_idx)
        plt.plot(range(n_generations), diversity_scores, label="Avg. Diversity", color="blue")
        plt.xlabel("Generation")
        plt.ylabel("% Diversity")
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

        plot_name = f"ID_{ID}_run_{test_run}_{problem_type}.png"
        os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, plot_name))
        plt.close()


    @staticmethod
    def record_experiment_results(file, run, ID, pop_size, trace_length, model, problem_type, mutation_type, termination_type,
                    exec_time, diversity_scores=None, constraint_scores=None, n_violations_scores=None, error=None):
        """
        Logs experiment results to a file.

        Parameters:
        ----------
        file : file
            The file where results are written.

        run : int
            The run number.

        ID : int
            The experiment identifier.

        pop_size : int
            Population size.

        trace_length : int
            Length of the traces.

        model : str
            The model used in the experiment.

        problem_type : str
            The type of problem being solved.

        exec_time : float
            Execution time of the experiment.

        diversity_scores : list, optional
            List of diversity scores over generations.

        constraint_scores : list, optional
            List of constraint scores over generations.

        n_violations_scores : list, optional
            List of violation scores over generations.

        error : str, optional
            Error message if an error occurred during execution.
        """

        status = f"{exec_time:.2f}" if exec_time else "ERROR"


        diversity_score = abs(diversity_scores[-1]) if diversity_scores else "NaN"
        diversity = f"{diversity_score:.2f}" if diversity_scores else "NaN"
        constraint = f"{constraint_scores[-1]:.2f}" if constraint_scores else "NaN"
        n_violations = f"{n_violations_scores[-1]:.2f}" if n_violations_scores else "NaN"


        file.write(
            f"{ID},{pop_size},{trace_length},{model},{problem_type.__name__},{type(mutation_type).__name__} eta={mutation_type.eta} prob ={mutation_type.prob.value},{termination_type},{status},"
            f"{diversity},{constraint},{run}\n"
        )

        if error:
            print(f"Error encountered with ID={ID}, Problem={problem_type.__name__}, Mutation:{type(mutation_type).__name__}: {error}")
        else:
            print(f"Execution Time ({problem_type.__name__}): {exec_time:.2f} seconds")

    @staticmethod
    def save_valid_solutions(population, encoder, run, ID, problem_type,
                             save_path, constraint_location, constraint_index=None):
        """
        Saves valid solutions from the final population to a file.

        Parameters:
        ----------
        population : list
            The final population from the genetic algorithm.

        encoder : Encoder object
            The encoder used to decode the traces.

        run : int
            The test run number.

        ID : int
            Experiment identifier.

        problem_type : str
            The type of problem being solved.

        constraint_location : str
            "G" for constraints evaluation for in G, "F" for constraints evaluation in F.

        constraint_index : int, optional
            Index of the constraint in F if constraint_location is "F".

        save_path : str
            Directory where the valid solutions will be saved.
        """

        final_population = [individual.X.tolist() for individual in population]

        # constraint values based on constraint_location
        if constraint_location == "G":
            constraints_array = np.array([individual.G for individual in population])
        else:  # constraint_location == "F"
            constraints_array = np.array([individual.F[constraint_index] for individual in population])

        # ensure directory exists
        os.makedirs(save_path, exist_ok=True)
        file_name = f"ID_{ID}_run_{run}_{problem_type}.csv"
        file_path = os.path.join(save_path, file_name)

        # feasible solutions (constraint should be <= 0)
        feasible_solutions = [trace for trace, constraint in zip(final_population, constraints_array) if
                              constraint <= 0]

        decoded_traces = encoder.decode(feasible_solutions)

        # save the traces into the file
        with open(file_path, "w") as f:
            for trace in decoded_traces:
                f.write(";".join(map(str, trace)) + "\n")
