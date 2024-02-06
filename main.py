import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_INSTRUCTIONS = 50
NUM_RESOURCES = 10
FITNESS_FACTOR = 1000000
TOURNAMENT_SIZE = 3


class Task:
    """
    Class representing a Task in the instruction
    """
    def __init__(self, resource, time):
        self.resource = resource
        self.time = time


class Operations:
    """
    Class representing an operations for single Job in the Excel file
    """
    def __init__(self, tasks):
        self.tasks = tasks


def read_operations(file_name='./GA_task.xlsx') -> dict[int, Operations]:
    """
    Read operations from the Excel file
    :param file_name: Name of the file
    :return: Dictionary with Operations from Excel file
    """
    df = pd.read_excel(file_name, skiprows=1)  # Skip the first row with the column names

    id = 1  # Operation id
    operations = {}  # Dictionary with Operations
    for i in range(0, len(df.columns), 2):  # Iterate through the columns with resources and times
        resources = df.iloc[:, i]  # Get the resources
        times = df.iloc[:, i + 1]  # Get the times
        jobs = []  # List with the tasks

        for j in range(0, len(resources)):  # Create tasks
            jobs.append(Task(resources[j], times[j]))

        operations[id] = Operations(jobs)
        id += 1

    return operations


operations = read_operations()


def calculate_time(order: list[int]) -> int:
    """
    Calculate the time of the order
    :param order: The order of operations
    :return: total time of the order
    """
    processing_times = {i: 0 for i in range(1, NUM_RESOURCES + 1)}  # Initialize resource processing times
    for _, operations_id in enumerate(order):
        current_operations = operations[operations_id]

        current_time = processing_times[current_operations.tasks[0].resource]  # Get the current time of the first resource

        for task in current_operations.tasks:  # Iterate through the tasks
            if current_time < processing_times[task.resource]:  # If the current time is less than the resource time
                current_time = processing_times[task.resource]  # Update the current time
            current_time += task.time  # Add the task time to the current time
            processing_times[task.resource] = current_time  # Update the resource processing time

    return max(processing_times.values())  # Return the maximum processing time


def fitness(order: list[int]) -> float:
    """
    Calculate the fitness of the order. The fitness is the inverse of the time to finish all tasks,
    multiplied by a factor to avoid very small numbers.
    :param order: Order of the instructions
    :return: Fitness of the order of operations
    """
    return 1 / calculate_time(order) * FITNESS_FACTOR


def generate_random_order() -> list[int]:
    """
    Generate a random order of operations
    :return: List with a random order of operations
    """
    order = list(range(1, NUM_INSTRUCTIONS + 1))
    random.shuffle(order)
    return order


def mutate_population(population: list[list[int]], mutation_rate: float):
    """
    Mutate the population by mutating each order
    :param population: The population to mutate
    :param mutation_rate: The mutation rate
    :return: None (the population is updated in place)
    """
    for order in population:
        if random.random() < mutation_rate:
            # Take 2 random indices and swap the elements at those indices
            index1, index2 = random.sample(range(len(order)), 2)
            order[index1], order[index2] = order[index2], order[index1]


def crossover(parents: list[list[int]], crossover_rate: float) -> list[list[int]]:
    """
    Perform crossover on the parents to generate offspring
    :param parents: List of parent orders
    :param crossover_rate: Probability of crossover
    :return: List with the offspring orders
    """
    n_parents = len(parents)
    offspring = []

    for i in range(0, n_parents, 2):
        parent1, parent2 = parents[i], parents[i + 1]

        if random.random() < crossover_rate:
            # Set the children to the first half of the parents
            child1, child2 = parent1[:len(parent1) // 2], parent2[:len(parent2) // 2]
            for j in parent2:
                if j not in child1:
                    child1.append(j)
            for j in parent1:
                if j not in child2:
                    child2.append(j)
        else:
            child1, child2 = parent1[:], parent2[:]

        offspring.extend([child1, child2])

    return offspring


def select_parents(population: list[list[int]], fitness_scores: list[float]) -> list[list[int]]:
    """
    Select parents for crossover using tournament selection. The parents are selected based on their fitness scores.
    :param population: The current population
    :param fitness_scores: The fitness scores of the population
    :return: The selected parents for crossover
    """
    selected_parents = []
    for _ in range(len(population)):
        # Select random individuals from the population
        tournament_indices = np.random.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]  # Get the fitness scores of the selected individuals
        winner_index = tournament_indices[np.argmax(tournament_fitness)]  # Select the winner of the tournament
        selected_parents.append(population[winner_index])  # Add the winner to the selected parents

    return selected_parents


def replace_population(current_population:  list[list[int]], offspring:  list[list[int]]) -> None:
    """
    Replace the current population with the offspring
    :param current_population: The current population
    :param offspring: The offspring
    :return: None (the current population is updated in place)
    """
    current_population.extend(offspring)
    current_population.sort(key=fitness, reverse=True)
    current_population[:] = current_population[:len(current_population) - len(offspring)]


def genetic_algorithm(n_iterations: int, population_size: int, crossover_rate: float, mutation_rate: float) -> list[int]:
    """
    Run the genetic algorithm and print the results
    :param n_iterations: The number of iterations
    :param population_size: The size of the population
    :param crossover_rate: The crossover rate
    :param mutation_rate: The mutation rate
    :return: List with the best order
    """
    population = [generate_random_order() for _ in range(population_size)]  # Generate the initial population
    best, best_score = population[0], fitness(population[0])  # Initialize the best order and score

    for _ in range(n_iterations):
        scores = [fitness(order) for order in population]  # Calculate the fitness of the population
        parents = select_parents(population, scores)  # Select the parents for crossover
        offspring = crossover(parents, crossover_rate)  # Generate the offspring
        mutate_population(offspring, mutation_rate)  # Mutate part of the offspring
        replace_population(population, offspring)  # Replace the population with the offspring

        for i in range(population_size):
            if scores[i] > best_score:
                best, best_score = population[i], scores[i]  # Update the best order and score

    print('Best order:\n', best)
    print('Best score:\n', best_score)
    print('Time:\n', calculate_time(best))
    return best


def create_gantt_chart(order) -> None:
    """
    Create and display Gantt chart from the given order
    :param order: The order from the genetic algorithm
    :return: None (the Gantt chart is displayed)
    """
    fig, ax = plt.subplots()

    # Initialize resource processing times
    res_proc = {i: 0 for i in range(1, NUM_RESOURCES + 1)}

    for _, operations_id in enumerate(order):
        current_operations = operations[operations_id]
        current_time = res_proc[current_operations.tasks[0].resource]

        for job in current_operations.tasks:
            if current_time < res_proc[job.resource]:
                current_time = res_proc[job.resource]
            ax.broken_barh([(current_time, job.time)], (job.resource * 10, 9), facecolors='tab:blue')
            current_time += job.time
            res_proc[job.resource] = current_time

    ax.set_xlabel('Time')
    ax.set_ylabel('Resource')
    ax.set_yticks([i * 10 + 5 for i in range(0, NUM_RESOURCES + 1)])
    ax.set_yticklabels(range(0, NUM_RESOURCES + 1))
    ax.grid(True)

    plt.show()


create_gantt_chart(range(1, NUM_INSTRUCTIONS + 1))  # Create a Gantt chart with the initial order

best_order = genetic_algorithm(10, 6, 0.25, 0.1)
create_gantt_chart(best_order)

best_order = genetic_algorithm(100, 60, 0.55, 0.2)
create_gantt_chart(best_order)

best_order = genetic_algorithm(500, 300, 0.8, 0.35)
create_gantt_chart(best_order)
