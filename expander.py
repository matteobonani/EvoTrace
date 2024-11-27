import random
import numpy as np

def expand_population(initial_population, expansion_factor, event_range):
    """
    Expand the initial population by generating variants of the input traces.
    """
    expanded_population = []

    for trace in initial_population:
        for _ in range(expansion_factor):

            new_trace = trace.copy()
            # Randomly replace or shuffle elements in the trace
            idx = random.randint(0, len(trace) - 1)
            new_trace[idx] = random.randint(event_range[0], event_range[1])
            expanded_population.append(new_trace)

    return expanded_population