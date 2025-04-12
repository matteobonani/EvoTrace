from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination
from ga_objects.callback import UpdatePopulationCallback
from utils.testSetup import Setup
from ga_objects.problem import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from ga_objects.sampling import MySampling
from ga_objects.terminator import MyTermination, DiversityTermination
from pymoo.algorithms.soo.nonconvex.ga import GA
import os
from typing import Dict, Any, Type

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------

def _get_scores_configuration(problem, constraint_scores, first_objective, second_objective) -> Dict[str, Any]:
    """Return the scoring configuration based on the problem type."""
    problem_type = type(problem)

    score_configs = {
        ProblemMultiElementWise: {
            "diversity_scores": first_objective,
            "constraint_scores": constraint_scores,
            "n_violations_scores": second_objective,
            "weighted_diversity_value": 1,
            "weighted_constraint_value": 1
        },
        ProblemMultiNoConstElementWise: {
            "diversity_scores": first_objective,
            "constraint_scores": second_objective,
            "n_violations_scores": None,
            "weighted_diversity_value": 0.2,
            "weighted_constraint_value": 0.8
        },
        ProblemSingleElementWise: {
            "diversity_scores": first_objective,
            "constraint_scores": constraint_scores,
            "n_violations_scores": None,
            "weighted_diversity_value": 1,
            "weighted_constraint_value": 1
        },
        ProblemSingle: {
            "diversity_scores": first_objective,
            "constraint_scores": constraint_scores,
            "n_violations_scores": None,
            "weighted_diversity_value": 1,
            "weighted_constraint_value": 1
        }
    }

    if problem_type not in score_configs:
        raise ValueError(f"Unsupported problem type for scoring configuration: {problem_type}")

    return score_configs[problem_type]



def _get_algorithm_configuration(problem: Any) -> Dict[str, Any]:
    """Return the algorithm configuration based on the problem type."""
    problem_type = type(problem)

    algorithm_configs = {
        ProblemMultiElementWise: {
            "algorithm": NSGA2,
            "constraint_location": "G",
            "constraint_index": None,
            "n_subplots": 3
        },
        ProblemMultiNoConstElementWise: {
            "algorithm": NSGA2,
            "constraint_location": "F",
            "constraint_index": 0,
            "n_subplots": 2
        },
        ProblemSingleElementWise: {
            "algorithm": GA,
            "constraint_location": "G",
            "constraint_index": None,
            "n_subplots": 2
        },
        ProblemSingle: {
            "algorithm": GA,
            "constraint_location": "G",
            "constraint_index": None,
            "n_subplots": 2
        }
    }

    if problem_type not in algorithm_configs:
        raise ValueError(f"Unsupported problem type for algorithm configuration: {problem_type}")

    return algorithm_configs[problem_type]



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

    d4py : str
        Path to the Declare model used for constraints.

    mutation : Mutation Object
        The mutation operator used in the genetic algorithm.

    crossover : Crossover Object
        The crossover operator used in the genetic algorithm.

    problem : Problem Class
        The class defining the optimization problem used to define all the proper configurations.

    termination : Termination Object
        The termination criteria for the algorithm.
    """

    def __init__(self, pop_size: int, trace_length: int, d4py: str, mutation: object, crossover: object, problem: Type[Problem], termination: object):
        self.pop_size = pop_size
        self.trace_length = trace_length
        self.d4py = d4py
        self.mutation = mutation
        self.crossover = crossover
        self.problem = problem
        self.termination = termination

    def _load_initial_population(self, model_name: str):
        """Loads the initial population based on the model name."""
        model_name_no_ext = os.path.splitext(model_name)[0]
        file_path = os.path.join(BASE_DIR, "..", "declare_models", model_name_no_ext, f"initial_pop_{self.trace_length}.csv")
        return Setup.extract_traces(file_path, self.trace_length)

    def _initialize_problem_instance(self, model_name: str, initial_population):
        """Initializes the problem instance."""
        model_path = os.path.join(BASE_DIR, "..", "declare_models", os.path.splitext(model_name)[0], model_name)

        encoder, d4py_obj, event_log, dataframe, encoded_pop, lb, ub = Setup.initialize_shared_components(
            path_to_declareModel=model_path,
            trace_length=self.trace_length,
            initial_population=initial_population
        )

        problem = self.problem(
            trace_length=self.trace_length,
            encoder=encoder,
            d4py=d4py_obj,
            initial_population=encoded_pop,
            xl=lb,
            xu=ub,
            event_log=event_log,
            dataframe=dataframe
        )

        return problem, encoder, encoded_pop

    def _setup_algorithm(self, problem_instance, encoded_pop):
        """Sets up the algorithm based on the problem type."""
        config = _get_algorithm_configuration(problem_instance)

        algorithm = config["algorithm"](
            problem=problem_instance,
            pop_size=self.pop_size,
            sampling=MySampling(initial_population=encoded_pop),
            crossover=self.crossover,
            mutation=self.mutation,
            callback=UpdatePopulationCallback(),
            eliminate_duplicates=False,
        )

        return algorithm, config

    def _handle_successful_run(self, population, n_gen, exec_time, problem_instance, encoder, run, ID, model_name, file, plot_path,
                               solution_path, algorithm_args, constraint_scores, first_obj, second_obj):
        """Handles post-processing, saving results, and plotting after a successful run."""
        scores_args = _get_scores_configuration(problem_instance, constraint_scores, first_obj, second_obj)

        diversity_scores, constraint_scores = Setup.invert_weighted_normalization(
            scores_args["diversity_scores"],
            scores_args["weighted_diversity_value"],
            scores_args["constraint_scores"],
            scores_args["weighted_constraint_value"]
        )

        Setup.record_experiment_results(file, run, ID, self.pop_size, self.trace_length, model_name,
                                        self.problem, self.mutation, self.termination, n_gen, exec_time,
                                        diversity_scores, constraint_scores, scores_args["n_violations_scores"])



        Setup.plot_evolution(ID, run, self.problem.__name__, algorithm_args["n_subplots"],
                             diversity_scores, n_gen, constraint_scores, scores_args["n_violations_scores"], plot_path)

        Setup.save_valid_solutions(population, encoder, run, ID, self.problem.__name__,
                                   solution_path, algorithm_args["constraint_location"],
                                   algorithm_args["constraint_index"])

    def _handle_failure(self, error, run, ID, model_name, file):
        """Handles failure and logs the error."""
        Setup.record_experiment_results(file, run, ID, self.pop_size, self.trace_length,
                                        model_name, self.problem, self.mutation,
                                        self.termination, 0, None, None, None, error=error)

    def run_experiment(self, run: int, ID: int, file: str, model_name: str, plot_path: str, solution_path: str):
        """
        Run the full experiment initializing, running the algorithm, and post-processing it.

        Parameters:
        ----------

        """
        try:
            # load initial population and initialize the problem
            initial_pop = self._load_initial_population(model_name)
            problem_instance, encoder, encoded_pop = self._initialize_problem_instance(model_name, initial_pop)

            # setup the optimization algorithm
            algorithm, algorithm_args = self._setup_algorithm(problem_instance, encoded_pop)

            # run
            constraint_scores, first_obj, second_obj, population, n_gen, exec_time = Setup.run_result(
                problem_instance, algorithm, self.termination, False
            )

            # save results, and plot
            self._handle_successful_run(population, n_gen, exec_time, problem_instance, encoder, run, ID,
                                        model_name, file, plot_path, solution_path,
                                        algorithm_args, constraint_scores, first_obj, second_obj)

        except Exception as e:
            # handle failure
            self._handle_failure(e, run, ID, model_name, file)