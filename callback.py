from pymoo.core.callback import Callback
import matplotlib.pyplot as plt
import numpy as np

# TODO plot la media della fit e score vincoli
class UpdatePopulationCallback(Callback):
    def __init__(self, problem, plot=1):
        super().__init__()
        self.problem = problem
        self.fitness_history = []
        self.constraint_scores = []
        self.generation = 0
        self.plot = plot

    def notify(self, algorithm):
        """
        Update the problem's current population and gather data for plotting.
        """
        # Update the current population
        population = algorithm.pop.get("X")
        self.problem.set_current_population(population)

        # Gather fitness and constraint scores for the current population
        fitness_scores = algorithm.pop.get("F")[:, 0]  # Assuming F[0] corresponds to constraint violations
        constraint_scores = algorithm.pop.get("G")[:, 0]  # Assuming G[0] is the constraint satisfaction score

        # Store the average scores for plotting
        self.fitness_history.append(np.mean(fitness_scores))
        self.constraint_scores.append(np.mean(constraint_scores))
        self.generation += 1

        if self.plot:
            self.plot_progress()

    def plot_progress(self):
        """
        Plot the average fitness and constraint scores over generations.
        """
        plt.figure(figsize=(10, 5))

        # Plot average constraint scores
        plt.subplot(1, 2, 1)
        plt.plot(range(self.generation), self.constraint_scores, label='Avg. Constraint Score', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Constraint Score')
        plt.title('Average Constraint Scores Over Generations')
        plt.grid(True)
        plt.legend()
        step_size = max(1, self.generation // 4)
        plt.xticks(range(0, self.generation, step_size))
        # plt.ylim(0, max(self.fitness_history) * 1.1)


        # Plot average fitness
        plt.subplot(1, 2, 2)
        plt.plot(range(self.generation), self.fitness_history, label='Avg. Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Average Fitness Over Generations')
        plt.grid(True)
        plt.legend()
        step_size = max(1, self.generation // 4)
        plt.xticks(range(0, self.generation, step_size))
        # plt.ylim(0, max(self.fitness_history) * 1.1)

        plt.tight_layout()
        # plt.pause(0.1)  # Pause for a brief moment to update the plots
        plt.show(block=False)

