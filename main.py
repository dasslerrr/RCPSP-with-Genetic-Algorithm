import csv
import random
class Activity:
    def __init__(self, identifier, resources, time, successors=None):
        self.identifier = identifier
        self.resources = resources
        self.time = time
        self.successors = successors if successors is not None else []

    def __str__(self):
        return f"Activity({self.identifier}, Resources: {self.resources}, Time: {self.time}, Successors: {self.successors})"

    def __repr__(self):
        return self.__str__()

def create_activities_from_csv(file_path):
    activities = {}
    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:  # Using utf-8-sig to handle BOM
        reader = csv.DictReader(file)
        # First pass: create all activities with empty successors
        for row in reader:
            job_number = row['jobnr'].strip()
            resources = int(row['resource'].strip())
            time = int(row['time'].strip())
            activities[job_number] = Activity(job_number, resources, time)

    # Second pass: update the successors
    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            job_number = row['jobnr'].strip()
            # Convert the string representation of the set to an actual set and then to a list
            successor_ids = eval(row['successors'].strip())
            activities[job_number].successors = [activities[str(succ)] for succ in successor_ids if str(succ) in activities]

    return list(activities.values())


def find_predecessors(activities):
    predecessors = {activity: set() for activity in activities}
    for activity in activities:
        for succ in activity.successors:
            predecessors[succ].add(activity)
    return predecessors

def generate_random_activity_sequence(activities):
    predecessors = find_predecessors(activities)
    available = set(activities)
    sequence = []

    while available:
        # Find activities that can be scheduled (all predecessors are already in the sequence)
        schedulable = [act for act in available if predecessors[act].issubset(sequence)]

        if not schedulable:
            raise ValueError("No feasible activity to schedule. There might be a circular dependency.")

        # Randomly select an activity from the schedulable ones and add it to the sequence
        next_activity = random.choice(schedulable)
        sequence.append(next_activity)
        available.remove(next_activity)

    return sequence

def print_activity_sequence(sequence):
    identifier_sequence = [activity.identifier for activity in sequence]
    print("Activity Sequence:", ' -> '.join(identifier_sequence))

def create_schedule(sequence, total_resources):
    total_time = 0
    current_resources = 0
    current_activities = []  # Activities running concurrently
    completed_activities = {}  # Dictionary to store completion time of activities
    start_times = {}  # Dictionary to store start times of activities

    # Calculate predecessors
    predecessors = find_predecessors(sequence)

    for activity in sequence:
        resources = activity.resources
        time = activity.time

        # Calculate the start time based on the completion time of the predecessors
        if predecessors[activity]:  # Check if there are any predecessors
            proposed_start_time = max(completed_activities.get(pred, 0) for pred in predecessors[activity])
        else:
            proposed_start_time = total_time

        # Check if current resources and the resources required by the activity do not exceed total resources
        if current_resources + resources <= total_resources and proposed_start_time >= total_time:
            if activity not in start_times:  # Set start time for new activities
                start_times[activity] = proposed_start_time
            current_resources += resources
            current_activities.append((activity, time))
        else:
            # Finish the current batch of activities and update total time
            if current_activities:
                max_time = max((act_time[1] for act_time in current_activities), default=0)
                total_time = max(total_time, max_time)

                for act, _ in current_activities:
                    completed_activities[act] = total_time
                    if act not in start_times:
                        start_times[act] = total_time - max_time

                current_resources = 0
                current_activities = []

        # Schedule the activity if it's not already scheduled and resources are available
        if activity not in start_times and current_resources + resources <= total_resources:
            proposed_start_time = max(total_time, max(completed_activities.get(pred, 0) for pred in predecessors[activity]))
            start_times[activity] = proposed_start_time
            current_resources += resources
            current_activities.append((activity, time))

    # Add the time for the last batch of activities
    if current_activities:
        max_time = max(current_activities, key=lambda x: x[1])[1]
        total_time += max_time
        for act, _ in current_activities:
            completed_activities[act] = total_time
            if act not in start_times:
                start_times[act] = total_time - max_time

    return {act.identifier: start for act, start in start_times.items()}


def print_schedule(schedule):
    for activity, start_time in schedule.items():
        print(f"Activity {activity.identifier} starts at time {start_time}")

def evaluate_schedule(schedule) -> int:
    return schedule[-1]

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


def print_schedule_formatted(schedule):
    formatted_output = ", ".join(f"({activity.identifier},{start_time})" for activity, start_time in schedule.items())
    print(formatted_output)

# Example usage
if __name__ == "__main__":
    # num_generations = 100
    # pop_size = 50
    # best_schedule, best_duration = genetic_algorithm(num_generations, pop_size)
    # print("Best schedule:", [activity for activity, _, _ in best_schedule])
    # print("Best duration:", best_duration)

    file_path = r"C:\Users\Gia-SanDang\Downloads\Citation\project_instances\instance1.csv"
    activities = create_activities_from_csv(file_path)

    sequence = generate_random_activity_sequence(activities)
    print_activity_sequence(sequence)

    schedule = create_schedule(sequence, 4)
    print_schedule_formatted(schedule)

