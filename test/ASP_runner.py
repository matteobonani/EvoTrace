import itertools
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.ProcessMiningTasks.LogGenerator.ASP.ASPLogGenerator import AspGenerator
import time
from ga_objects.utils import Encoder
from scipy.spatial.distance import pdist
import numpy as np
from ga_objects.utils.tools import Tools

from datetime import datetime

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
    declare_model_list = ["model1","model2","model3","model4"]

    # timestamped results directory
    current_date = datetime.today().strftime('%m-%d-%H-%M')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, "results", current_date)
    os.makedirs(directory, exist_ok=True)

    file_name = os.path.join(directory, "result.csv")

    with open(file_name, "a") as f:
        f.write("ID,Run,Model,PopulationSize,TraceLength,ExecTime,Diversity\n")

        for run in range(1, num_runs + 1):
            ID = 1
            for combination in itertools.product(pop_list, num_event_list, declare_model_list):
                pop_size, trace_length, model_name = combination


                file_dir = os.path.join(base_dir,"..", "declare_models", f"{model_name}",f"{model_name}.decl")
                model: DeclareModel = DeclareModel().parse_from_file(file_dir)
                (num_min_events, num_max_events) = (trace_length, trace_length)
                verbose = False

                asp_gen: AspGenerator = AspGenerator(model, pop_size, num_min_events, num_max_events, verbose=verbose)

                start_time = time.time()
                asp_gen.run()
                end_time = time.time()
                exec_time = end_time - start_time

                asp_gen.to_xes(f"ASP")

                declare = DeclareModel().parse_from_file(file_dir)
                activities = declare.activities
                encoder = Encoder(activities)

                traces = Tools.extract_traces_from_xes("ASP.xes", trace_length)

                encoded_traces = [encoder.encode(trace) for trace in traces]

                population = np.array(encoded_traces)

                dist_matrix = pdist(population, metric='hamming')

                f.write(f"{ID},{run},{model_name},{pop_size},{trace_length},{exec_time:.4f}s,{dist_matrix.mean():.6f}\n")

                ID += 1


if __name__ == "__main__":
    main()
