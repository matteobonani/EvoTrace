class DataCollector:
    def __init__(self):
        self.fitness_history = []
        self.constraint_history = []
        self.diversity_history = []
        self.generation = []

    def update(self, generation, fitness, constraints, diversity=None):

        print("updating data")
        self.fitness_history.append(fitness)
        self.constraint_history.append(constraints)
        if diversity is not None:
            self.diversity_history.append(diversity)
        self.generation.append(generation)


    def get_data(self):
        return {
            "fitness_history": self.fitness_history,
            "constraint_history": self.constraint_history,
            "diversity_history": self.diversity_history,
            "generations": self.generation,
        }