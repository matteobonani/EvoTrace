from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination

from ga_objects.callback import UpdatePopulationCallback
from utils.testSetup import Setup
from ga_objects.problem import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from ga_objects.sampling import MySampling
from ga_objects.terminator import MyTermination, DiversityTermination
from pymoo.algorithms.soo.nonconvex.ga import GA
import os


# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------

# def get_initial_population(model_name, n_event_per_trace):
#     config = {
#         "model1.csv": {
#             "initial_population": get_initial_population_length(n_event_per_trace)
#         },
#         "model2.csv": {
#             "diversity_scores": first_objective
#         },
#         "model3.csv": {
#             "diversity_scores": first_objective
#         },
#         "model4.csv": {
#             "diversity_scores": first_objective
#         }
#     }
#
#     return config.get(model_name, {})
#
# def get_initial_population_length(n_event):



def get_result_scores_arguments(problem, constraint_scores, first_objective, second_objective):
    """Determines the appropriate scoring arguments based on the type of problem."""
    problem_type = type(problem)

    config = {
        ProblemMultiElementWise: {
            "diversity_scores": first_objective,
            "constraint_scores": constraint_scores,
            "n_violations_scores": second_objective,
            "weighted_diversity_value":1,
            "weighted_constraint_value":1
        },
        ProblemMultiNoConstElementWise: {
            "diversity_scores": first_objective,
            "constraint_scores": second_objective,
            "n_violations_scores": None,
            "weighted_diversity_value":0.2,
            "weighted_constraint_value":0.8
        },
        ProblemSingleElementWise: {
            "diversity_scores": first_objective,
            "constraint_scores": constraint_scores,
            "n_violations_scores": None,
            "weighted_diversity_value":1,
            "weighted_constraint_value":1
        },
        ProblemSingle: {
            "diversity_scores": first_objective,
            "constraint_scores": constraint_scores,
            "n_violations_scores": None,
            "weighted_diversity_value": 1,
            "weighted_constraint_value": 1
        }
    }

    return config.get(problem_type, {})


def get_result_algorithm_arguments(problem, termination, pop_size):
    """Determines the algorithm configuration based on the problem type."""
    problem_type = type(problem)

    config = {
        ProblemMultiElementWise: {
            "algorithm": NSGA2,
            "termination": get_terminator(problem_type, termination, pop_size, "F", 1),
            "constraint_location": "G",
            "constraint_index": None,
            "n_subplots": 3
        },
        ProblemMultiNoConstElementWise: {
            "algorithm": NSGA2,
            "termination": get_terminator(problem_type, termination, pop_size, "F", 1),
            "constraint_location": "F",
            "constraint_index": 0,
            "n_subplots": 2
        },
        ProblemSingleElementWise: {
            "algorithm": GA,
            "termination": get_terminator(problem_type, termination, pop_size, "G", 0),
            "constraint_location": "G",
            "constraint_index": None,
            "n_subplots": 2
        },
        ProblemSingle:{
            "algorithm": GA,
            "termination": get_terminator(problem_type, termination, pop_size, "G", 0),
            "constraint_location": "G",
            "constraint_index": None,
            "n_subplots": 2
        }
    }

    algorithm_config = config.get(problem_type, {})

    if not algorithm_config:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return algorithm_config


def get_terminator(problem_type, termination, pop_size, constraint_location, constraint_index):
    """Determines the termination condition for the optimization algorithm based on the problem type and user-defined termination settings."""
    if termination == "myTerminator":
        # return MyTermination(pop_size, constraint_location, constraint_index, 300)
        return DiversityTermination(0.9, 200)
    # default terminations based on problem type
    if problem_type in {ProblemMultiElementWise, ProblemMultiNoConstElementWise}:
        return DefaultMultiObjectiveTermination(
            xtol=1e-8, cvtol=1e-6, ftol=0.0025, period=30,
            n_max_gen=1000, n_max_evals=100000
        )
    elif problem_type == ProblemSingleElementWise:
        return DefaultSingleObjectiveTermination(
            xtol=1e-6, ftol=0.001, period=20,
            n_max_gen=500, n_max_evals=50000
        )
    else:
        raise ValueError("Unknown problem type. Cannot determine termination.")


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------

class ProblemSetup:
    """
    Handles the setup and execution of an optimization experiment using the appropriate settings for each problem.

    Parameters:
    ----------
    pop_size : int
        The size of the population in the algorithm.

    trace_length : int
        The length of traces in the optimization problem.

    d4py_path : str
        Path to the Declare model used for constraints.

    mutation : Mutation Object
        The mutation operator used in the genetic algorithm.

    crossover : Crossover Object
        The crossover operator used in the genetic algorithm.

    problem : Problem Class
        The class defining the optimization problem used to define all the proper configurations.

    initial_population : list or array
        The initial population in the optimization problem.
    """
    def __init__(self, pop_size, trace_length, d4py, mutation, crossover, problem, termination):
        self.pop_size = pop_size
        self.trace_length = trace_length
        self.d4py = d4py
        self.mutation = mutation
        self.crossover = crossover
        self.problem = problem
        self.termination = termination



    def run_experiment(self, run, ID, file, model_name, plot_path, solution_path):
        """
        Executes a single optimization and save the solutions with the specified configuration.

        Parameters:
        ----------
        run : int
            The current run number of the experiment.

        ID : str
            A unique identifier for the experiment.

        file : str
            Path to the file where the results should be recorded.

        model_name : str
            The name of the model being used in the experiment.

        plot_path : str
            Path where the plots showing the experiment's evolution should be saved.

        solution_path : str
            Path to save the valid solutions found during the experiment.
        """

        # model_name_without_ext = os.path.splitext(model_name)[0]
        # if initial_population_type == "feasible":
        #     initial_population = Setup.initial_population(f"../declare_models/{model_name_without_ext}_initial_pop.csv",
        #                                                   trace_length=self.trace_length, trace_number=10)
        # else:
        #     initial_population = Setup.initial_random_population(f"../declare_models/{model_name}",
        #                                                          trace_length=self.trace_length, trace_number=10)

        model_name_without_ext = os.path.splitext(model_name)[0]
        file_path = os.path.join("..", "declare_models", model_name_without_ext,
                                 f"initial_pop_{self.trace_length}.csv")
        initial_population = Setup.extract_traces(file_path, self.trace_length)

        # initialize shared components for problem setup
        encoder, d4py, event_log, dataframe, initial_encoded_pop, lower_bounds, upper_bounds = Setup.initialize_shared_components(
            path_to_declareModel=f"../declare_models/{model_name_without_ext}/{model_name}",
            trace_length=self.trace_length,
            initial_population=initial_population,
            pop_size=self.pop_size
        )

        # initialize the problem based on dynamic arguments
        problem_instance = self.problem(
            trace_length=self.trace_length,
            encoder=encoder,
            d4py=d4py,
            initial_population=initial_encoded_pop,
            xl=lower_bounds,
            xu=upper_bounds,
            event_log=event_log,
            dataframe=dataframe,
        )

        algorithm_args = get_result_algorithm_arguments(problem_instance, self.termination, self.pop_size)

        # initialize the algorithm based on the arguments
        algorithm = algorithm_args["algorithm"](
            problem=problem_instance,
            pop_size=self.pop_size,
            sampling=MySampling(initial_population=initial_encoded_pop),
            crossover=self.crossover,
            mutation=self.mutation,
            callback=UpdatePopulationCallback(),
            eliminate_duplicates=False,
        )



        # terminator = MyTermination(self.pop_size, algorithm_args["constraint_location"], algorithm_args["constraint_index"], 1200)

        try:

            exec_time, constraint_scores, first_objective, second_objective, n_generations, population = (
                Setup.run_result(problem_instance, algorithm, algorithm_args["termination"], False))



            scores_args = get_result_scores_arguments(problem_instance, constraint_scores, first_objective, second_objective)

            diversity_scores, constraint_scores = Setup.invert_wighted_normalization(scores_args["diversity_scores"], scores_args["weighted_diversity_value"],
                                               scores_args["constraint_scores"] ,scores_args["weighted_constraint_value"], self.trace_length)


            Setup.record_experiment_results(file, run, ID, self.pop_size, self.trace_length, model_name,
                                            self.problem, self.mutation, self.termination, exec_time, diversity_scores, constraint_scores,
                                            scores_args["n_violations_scores"])


            Setup.plot_evolution(ID, run, self.problem.__name__, algorithm_args["n_subplots"], diversity_scores,
                                  constraint_scores, scores_args["n_violations_scores"], n_generations, plot_path)


            Setup.save_valid_solutions(population, encoder, run, ID, self.problem.__name__, solution_path,
                                       algorithm_args["constraint_location"], algorithm_args["constraint_index"])

        except Exception as algo_error:
            Setup.record_experiment_results(file, run, ID, self.pop_size, self.trace_length, model_name,
                                            self.problem, self.mutation, self.termination,0, None, None, None, error=algo_error)
