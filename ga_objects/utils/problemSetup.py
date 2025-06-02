from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from ga_objects.operators.callback import SingleObjectiveCallback, MultiObjectiveCallback
from ga_objects.utils.testSetup import Setup
from ga_objects.problems.single_objective_problems import *
from ga_objects.problems.multi_objective_problems import *
from ga_objects.operators.sampling import MySampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
import os
from typing import Dict, Any, Type

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------

def _get_scores_configuration(problem, data) -> Dict[str, Any]:
    """Return the scoring configuration based on the problem type."""
    problem_type = type(problem)

    score_configs = {
        ProblemMulti: {
            "diversity_scores": data.get("first_objective"),
            "constraint_scores": data.get("second_objective"),
            "n_violations_scores": None,
            "weighted_diversity_value": 1,
            "weighted_constraint_value": 1
        },
        ProblemMultiObjectiveNovelty: {
            "diversity_scores": data.get("first_objective"),
            "constraint_scores": data.get("second_objective"),
            "n_violations_scores": None,
            "weighted_diversity_value": 1,
            "weighted_constraint_value": 1
        },
        ProblemSingle: {
            "diversity_scores": data.get("first_objective"),
            "constraint_scores": data.get("constraint_history"),
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
        ProblemMulti: {
            "algorithm": RNSGA2,
            "callback": MultiObjectiveCallback(),
            "constraint_location": "F",
            "constraint_index": 1,
            "n_subplots": 2
        },
        ProblemMultiObjectiveNovelty: {
            "algorithm": NSGA3,
            "callback": MultiObjectiveCallback(),
            "constraint_location": "G",
            "constraint_index": None,
            "n_subplots": 2
        },
        ProblemSingle: {
            "algorithm": GA,
            "callback": SingleObjectiveCallback(),
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
        file_path = os.path.join(BASE_DIR, "..", "..", "declare_models", model_name_no_ext, f"initial_pop_{self.trace_length}.csv")
        return Setup.extract_traces(file_path, self.trace_length)

    def _initialize_problem_instance(self, model_name: str, initial_population):
        """Initializes the problem instance."""
        model_path = os.path.join(BASE_DIR, "..","..", "declare_models", os.path.splitext(model_name)[0], model_name)

        encoder, d4py_obj, event_log, dataframe, encoded_pop, lb, ub = Setup.initialize_shared_components(
            path_to_declareModel=model_path,
            trace_length=self.trace_length,
            initial_population=initial_population
        )

        self.problem = self.problem(
            trace_length=self.trace_length,
            encoder=encoder,
            d4py=d4py_obj,
            initial_population=encoded_pop,
            xl=lb,
            xu=ub,
            event_log=event_log,
            dataframe=dataframe
        )

        return encoder, encoded_pop, d4py_obj

    def _setup_algorithm(self, encoded_pop):
        """Sets up the algorithm based on the problem type."""
        config = _get_algorithm_configuration(self.problem)

        algorithm = config["algorithm"](
            problem=self.problem,
            pop_size=self.pop_size,
            sampling=MySampling(initial_population=encoded_pop),
            crossover=self.crossover,
            mutation=self.mutation,
            callback=config["callback"],
            eliminate_duplicates=False,
            ref_points = np.array([[-0.7, 0.05]]),
            extreme_points_as_reference_points=False,
        )

        return algorithm, config

    def _handle_successful_run(self, d4py, population, n_gen, exec_time, encoder, run, ID, model_name, file, plot_path,
                               solution_path, algorithm_args, data):
        """Handles post-processing, saving results, and plotting after a successful run."""
        scores_args = _get_scores_configuration(self.problem, data)

        diversity_scores, constraint_scores = Setup.invert_weighted_normalization(
            scores_args["diversity_scores"],
            scores_args["weighted_diversity_value"],
            scores_args["constraint_scores"],
            scores_args["weighted_constraint_value"],
            len(d4py.get_decl_model_constraints()),
            data.get("max_diversity", None)
        )

        Setup.record_experiment_results(file, run, ID, self.pop_size, self.trace_length, model_name,
                                        self.problem, self.mutation, self.termination, n_gen, exec_time,
                                        diversity_scores, constraint_scores, scores_args["n_violations_scores"])

        Setup.plot_evolution(ID, run, type(self.problem).__name__, algorithm_args["n_subplots"],
                             diversity_scores, n_gen, constraint_scores, scores_args["n_violations_scores"], plot_path)

        Setup.save_valid_solutions(population, encoder, run, ID, type(self.problem).__name__,
                                   solution_path, algorithm_args["constraint_location"],
                                   algorithm_args["constraint_index"])

    def _handle_failure(self, error, run, ID, model_name, file):
        """Handles failure and logs the error."""
        Setup.record_experiment_results(file, run, ID, self.pop_size, self.trace_length,
                                        model_name, self.problem, self.mutation,
                                        self.termination, 0, None, None, None, error=error)

    def run(self, run: int, ID: int, file: str, model_name: str, plot_path: str, solution_path: str):
        """
        Run the full experiment initializing, running the algorithm, and post-processing it.

        """
        try:
            # load initial population and initialize the problem
            initial_pop = self._load_initial_population(model_name)

            encoder, encoded_pop, d4py = self._initialize_problem_instance(model_name, initial_pop)

            # setup the optimization algorithm
            algorithm, algorithm_args = self._setup_algorithm(encoded_pop)

            # run
            data, population, n_gen, exec_time = Setup.run_result(
                self.problem, algorithm, self.termination, False
            )

            # save results, and plot
            self._handle_successful_run(d4py, population, n_gen, exec_time, encoder, run, ID,
                                        model_name, file, plot_path, solution_path,
                                        algorithm_args, data)

        except Exception as e:
            # handle failure
            self._handle_failure(e, run, ID, model_name, file)