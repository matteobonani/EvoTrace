import numpy as np

class Encoder:
    def __init__(self, activities):
        self.activities = np.array(activities)
        self.activity_to_index = {act: idx for idx, act in enumerate(activities)}
        self.index_to_activity = np.array(activities)


    def encode(self, traces):
        traces = np.array(traces)

        if traces.ndim == 1:
            return np.vectorize(self.activity_to_index.get)(traces)
        else:
            return np.array([[self.activity_to_index[act] for act in trace] for trace in traces], dtype=int)

    def decode(self, encoded_traces):

        encoded_traces = np.array(encoded_traces)

        if encoded_traces.ndim == 1:
            return self.index_to_activity[encoded_traces]
        else:
            return np.array([[self.index_to_activity[idx] for idx in trace] for trace in encoded_traces], dtype=object)

