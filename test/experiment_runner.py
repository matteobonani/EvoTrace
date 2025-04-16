import warnings
import logging
import itertools
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from utils.problemSetup import ProblemSetup
from ga_objects.problem import ProblemSingle
from pymoo.operators.crossover.sbx import SBX
from ga_objects.mutation import IntegerPolynomialMutation
from ga_objects.terminator import DiversityTermination
from pymoo.operators.crossover.pntx import TwoPointCrossover

# silence warnings and logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*feasible.*")


def main():

    try:
        num_runs = int(input("Enter number of runs: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    print(f"Starting {num_runs} run of optimization...")

    # configuration lists
    pop_list = [1000,2000,3000,4000]
    num_event_list = [30,50,70,90]
    declare_model_list = ["model1.decl", "model2.decl", "model3.decl", "model4.decl"]
    mutation_list = [IntegerPolynomialMutation(prob=0.5, eta=1)]
    crossover_list = [TwoPointCrossover(prob=0.9)]
    problem_list = [ProblemSingle]
    termination_list = [DiversityTermination(0.9, 300)]

    # timestamped results directory
    current_date = datetime.today().strftime('%m-%d-%H-%M')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, "results", current_date)
    os.makedirs(directory, exist_ok=True)

    file_name = os.path.join(directory, "result.csv")

    with open(file_name, "a") as f:
        f.write("ID,Population,TraceLength,Model,Problem,Mutation,Termination,"
                "ExecutionTime,DiversityScore,ConstraintScore,Generations,Iteration\n")

        for run in range(1, num_runs + 1):
            ID = 1
            for combination in itertools.product(pop_list, num_event_list, declare_model_list,
                                                 mutation_list, crossover_list,
                                                 problem_list, termination_list):
                pop_size, trace_length, model, mutation, crossover, problem, termination = combination

                print(f"Run {run} - ID {ID}: Population={pop_size}, TraceLength={trace_length}, Model={model}, "
                      f"Problem={problem.__name__}, Mutation={type(mutation).__name__}, "
                      f"Crossover={type(crossover).__name__}, Termination={type(termination).__name__}")

                plot_path = os.path.join(base_dir, "results", f"{directory}/plots")
                solution_path = os.path.join(base_dir, f"{directory}/encoded_traces")

                problemSetup = ProblemSetup(pop_size, trace_length, model, mutation, crossover, problem, termination)
                problemSetup.run_experiment(run, ID, f, model, plot_path, solution_path)

                ID += 1


if __name__ == "__main__":
    main()
