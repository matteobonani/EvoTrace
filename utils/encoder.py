import numpy as np

class Encoder:
    """
    A class for encoding and decoding activity traces using NumPy arrays for efficiency.

    Parameters:
    ----------
    activities : list of str
        A list containing the activity names.
    """
    def __init__(self, activities):
        self.activities = np.array(activities)
        self.activity_to_vector = {activity: idx for idx, activity in enumerate(activities)}
        self.vector_to_activity = np.array(activities)

    def encode(self, traces):
        """Convert a list of activity sequences into their numerical representation."""

        traces = np.array(traces, dtype=object)


        if traces.ndim == 1:  # 1D array
            return [self.activity_to_vector[activity] for activity in traces]
        else:  # 2D array
            return [[self.activity_to_vector[activity] for activity in trace] for trace in traces]

    def decode(self, encoded_traces):
        """Convert encoded numerical sequences back into their original activity sequences."""



        encoded_traces = np.array(encoded_traces, dtype=object)

        if encoded_traces.ndim == 1:  # 1D array
            return [self.vector_to_activity[idx] for idx in encoded_traces]
        else:  # 2D array
            return [[self.vector_to_activity[activity] for activity in trace] for trace in encoded_traces]
