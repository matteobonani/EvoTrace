class Encoder:
    """
    A class for encoding and decoding activity traces.

    This class provides methods to convert activity sequences into numerical representations and back.

    Parameters:
    ----------
    activities : list of str
        A list containing the activity names.
    """
    def __init__(self, activities):
        self.activities = activities
        self.activity_to_vector = {activity: idx for idx, activity in enumerate(activities)}
        self.vector_to_activity = {idx: activity for idx, activity in enumerate(activities)}


    def encode(self, trace):
        """Convert an activity sequence into its numerical representation."""

        return [self.activity_to_vector[activity] for activity in trace]

    def decode(self, encoded_trace):
        """Convert an encoded numerical sequence back into its original activity sequence."""

        return [self.vector_to_activity[idx] for idx in encoded_trace]

