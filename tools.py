
import random
from datetime import datetime, timedelta
import numpy as np

class Tools:

    @staticmethod
    def calculate_feature_range(population, variable_boundaries):
        """
        Calculate the feature range for each variable across the entire population.
        """

        population = np.array(population)  # Convert population to a NumPy array for faster operations
        num_variables = population.shape[1]  # Number of features/variables

        feature_ranges = []

        for j in range(num_variables):

            if variable_boundaries[j] == 1:  # non-dummy variable
                min_val = np.min(population[:, j])
                max_val = np.max(population[:, j])
                feature_ranges.append((min_val, max_val))
            else:  # dummy variable, range is always [0, 1]
                feature_ranges.append((0, 1))

        return feature_ranges

    @staticmethod
    def expand_population(initial_population, final_population_size, feature_range):
        """
        Expands the population by modifying the given traces until the target size is reached.
        """

        final_population = initial_population.copy()

        while len(final_population) < final_population_size:
            for trace in initial_population:

                new_trace = trace.copy()

                # TODO better to not modify more than once a dummy variable, and also check that inside the range, only one variable can be set as 1 [check mutations]
                # randomly select the number of columns to change
                num_columns_to_change = random.randint(1, len(feature_range))

                # selected the columns to modify in the trace
                columns_to_modify = random.sample(range(len(feature_range)), num_columns_to_change)

                for column_idx in columns_to_modify:

                    min_value, max_value = feature_range[column_idx]
                    new_value = random.randint(min_value, max_value)

                    new_trace[column_idx] = new_value


                final_population.append(new_trace)

        return final_population

    @staticmethod
    def generate_random_timestamps(length):
        start_date = datetime(2024, 12, 10, 10, 0, 0)
        timestamps = []

        for _ in range(length):
            random_minutes = random.randint(1, 10)
            start_date += timedelta(minutes=random_minutes)
            timestamps.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))

        return timestamps