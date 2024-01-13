import matplotlib.pyplot as plt

# Maximum resource level
max_resource_level = 4


# Define the rectangles information array with the provided tuples
rectangles_info = [
    (0, 0, 3, 2, '1'),
    (3, 0, 2, 1, '2'),
    (3, 1, 1, 2, '3'),
    (4, 1, 1, 3, '4'),
    (5, 0, 1, 4, '6')
]

# Initialize the figure and axes
fig, ax = plt.subplots()

# Set the limits and labels of the axes
ax.set_xlim(0, 14)
ax.set_ylim(0, max_resource_level + 1)
ax.set_xlabel('Time')
ax.set_ylabel('Resources')
ax.set_yticks(range(1, max_resource_level + 1))

# Draw a horizontal line to represent the maximum resource level
ax.hlines(max_resource_level, 0, 14, colors='black', linestyles='dotted', lw=2)


def draw_rectangles(ax, rectangles_info):
    """
    Draw multiple rectangles on the given axes.

    :param ax: Matplotlib axes object where rectangles will be drawn.
    :param rectangles_info: List of tuples, where each tuple contains:
                            (x, y, width, height, 'text')
    """
    for (x, y, width, height, text) in rectangles_info:
        # Draw the rectangle
        rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        # Place the text in the center of the rectangle
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center')

draw_rectangles(ax, rectangles_info)


# Show the plot
plt.show()


def crossover(parent1, parent2):
    # Order-based crossover (OX)
    # Choose two random crossover points
    crossover_point1 = random.randint(0, len(parent1) - 1)
    crossover_point2 = random.randint(crossover_point1 + 1, len(parent1))

    # Create two empty offspring with the same length as the parents
    offspring1 = [None] * len(parent1)
    offspring2 = [None] * len(parent1)

    # Copy the genetic material between the crossover points from parent1 to offspring1 and from parent2 to offspring2
    offspring1[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]
    offspring2[crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]

    # Fill in the remaining positions with genetic material from parent2 to offspring1 and from parent1 to offspring2
    idx2 = crossover_point2
    idx1 = crossover_point2
    while None in offspring1:
        gene2 = parent2[idx2 % len(parent2)]
        gene1 = parent1[idx1 % len(parent1)]
        if gene2 not in offspring1:
            offspring1[idx1 % len(parent1)] = gene2
        if gene1 not in offspring2:
            offspring2[idx2 % len(parent2)] = gene1
        idx1 += 1
        idx2 += 1

    return offspring1, offspring2


def genetic_algorithm(num_generations, pop_size):
    population = [generate_random_schedule() for _ in range(pop_size)]

    for generation in range(num_generations):
        # Evaluate fitness of the population
        fitness_scores = [evaluate_schedule(schedule) for schedule in population]

        # Create new generation through tournament selection, crossover, and mutation
        new_population = []
        while len(new_population) < pop_size:
            # Tournament selection (select two random schedules and pick the best)
            tournament_size = min(5, len(population))
            tournament = random.sample(list(enumerate(population)), tournament_size)
            tournament.sort(key=lambda x: fitness_scores[x[0]])
            parent1 = tournament[0][1]
            parent2 = tournament[1][1]

            offspring1, offspring2 = crossover(parent1, parent2)

            # Mutation
            if random.random() < 0.1:
                offspring1 = generate_random_schedule()
            if random.random() < 0.1:
                offspring2 = generate_random_schedule()

            # Add offspring to the new population
            new_population.extend([offspring1, offspring2])

        # Sort the population by fitness
        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1])
        population = [schedule for schedule, _ in population_with_fitness[:pop_size]]

    # Select the best solution as the result
    best_schedule = min(population, key=evaluate_schedule)
    return best_schedule, evaluate_schedule(best_schedule)