import csv
import itertools
import random
import glob
import time

import matplotlib.pyplot as plt

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


import csv

def create_activities_from_csv(file_path):
    activities = {}
    total_resource = 0
    alternative_chains = []

    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        # Use csv.reader to handle rows uniformly
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Check if the row is total_resource or alternative_chain
            if row[0].lower() == 'total_resource':
                total_resource = int(row[1])
            elif row[0].lower() == 'alternative_chain':
                break  # Exit loop after total_resource

        for row in csv_reader:  # Continue reading for alternative_chain
            if row[0] == '---':  # Check for delimiter row
                break
            else:
                # Process and add the alternative pairs
                pair = eval(row[1])
                pair = (str(pair[0]).strip("'"), str(pair[1]).strip("'"))
                alternative_chains.append((str(pair[0]), str(pair[1])))

        for row in csv_reader:  # Continue reading for job details
            if row[0].lower() == 'jobnr':  # Skip header row
                continue
            job_number = row[0].strip("'")
            resources = int(row[1].strip())
            time = int(row[2].strip())
            activities[job_number] = Activity(job_number, resources, time)

    # Second pass for successors, re-open file to avoid resetting csv_reader
    with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0].lower() == 'jobnr':  # Find the start of the job details
                break

        for row in csv_reader:
            job_number = row[0].strip("'")
            if job_number in activities:  # Check if job_number exists in activities
                successor_ids = eval(row[3].strip())
                successor_ids = tuple(element.strip("'") if isinstance(element, str) else element for element in successor_ids)
                activities[job_number].successors = [activities[str(succ)] for succ in successor_ids if str(succ) in activities]

    return total_resource, list(activities.values()), alternative_chains

def find_predecessors(activities):
    predecessors = {activity: set() for activity in activities}
    for activity in activities:
        for succ in activity.successors:
            if succ in activities:
                predecessors[succ].add(activity)
    return predecessors

def generate_activity_sequence(activities, activity_list):
    sequence = []
    for item in activity_list:
        for act in activities:
            if str(item) == act.identifier:
                sequence.append(act)
                break
    return sequence

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

def remove_activity_by_identifier(activities, identifier):
    temp = activities.copy()
    for act in temp:
        if act.identifier == identifier:
            temp.remove(act)
            return temp

def create_schedule(activity_sequence, total_resources):
    """
    Creates a schedule for a given sequence of activities.

    :param activity_sequence: List of Activity objects in the order they should be scheduled.
    :param total_resources: The total available resources.
    :return: A list of tuples, each containing an Activity object and its start time.
    """
    predecessors = find_predecessors(activity_sequence)
    scheduled = []
    finish_time = [0]

    for activity in activity_sequence:
        finish_time.sort()
        for current_time in finish_time:
            if is_precedence_feasible(scheduled, current_time, activity, predecessors):
                if is_resource_feasible(scheduled, current_time, activity, total_resources):
                    scheduled.append((activity, current_time))
                    finish_time.append(current_time + activity.time)
                    break
        finish_time = optimize_finish_time(scheduled, finish_time, total_resources)
    return scheduled

def optimize_finish_time(scheduled, finish_time, total_resource):
    for time in finish_time:
        resource = 0
        for act_tuple in scheduled:
            if act_tuple[1] <= time < act_tuple[1] + act_tuple[0].time:
                resource += act_tuple[0].resources
        if resource == total_resource:
            finish_time.remove(time)
    return finish_time

def is_resource_feasible(scheduled, time, activity, total_resource):
    end_time = time + activity.time
    while time < end_time:
        resource = 0
        for act_tuple in scheduled:
            if act_tuple[1] <= time < act_tuple[1] + act_tuple[0].time:
                resource += act_tuple[0].resources
        if activity.resources + resource > total_resource: return False
        time += 1
    return True

def is_precedence_feasible(scheduled, time, activity, predecessors):
    # No predecessors -> True
    scheduled_activities = [item[0] for item in scheduled]
    if predecessors[activity] is None:
        return True
    else:
        # Check for each predecessor
        for pred in predecessors[activity]:
            for activity_tupel in scheduled:
                if activity_tupel[0] == pred:
                    if activity_tupel[1] + pred.time > time:
                        return False
    return True

def print_activity_sequence(sequence):
    identifier_sequence = [activity.identifier for activity in sequence]
    print("Activity Sequence:", ' -> '.join(identifier_sequence))

def print_schedule_formatted(schedule):
    formatted_output = ", ".join(f"({activity.identifier},{start_time})" for activity, start_time in schedule)
    print("Schedule: " + formatted_output)

def draw_schedule(schedule, total_resources):
    schedule.sort(key=lambda x: (x[1], -x[0].time))

    # Initialize the 2D array for resource tracking
    resource_grid = [[0 for _ in range(total_resources)] for _ in
                     range(max(schedule, key=lambda x: x[1])[1] + max(schedule, key=lambda x: x[0].time)[0].time)]

    total_time = schedule[-1][1] + schedule[-1][0].time
    # Function to find space for an activity
    def find_space_for_activity(activity_duration, activity_resources, schedule_start_time):
        for resource_level in range(total_resources - activity_resources + 1):
            if all(resource_grid[schedule_start_time + t][resource_level + r] == 0 for t in range(activity_duration) for r in
                   range(activity_resources)):
                return resource_level
        return None

    def colision_activity(scheduled, activity, start_time):
        for activity_2, start_time_2, level in reversed(scheduled):
            activity1_end_time = start_time + activity.time
            activity2_end_time = start_time_2 + activity_2.time

            overlap = (start_time < activity2_end_time and activity1_end_time > start_time_2) or \
                      (start_time_2 < activity1_end_time and activity2_end_time > start_time)

            if overlap:
                return (activity_2, start_time_2, level)
    # Function to update the resource grid
    def update_resource_grid(scheduled):
        temp = [[0 for _ in range(total_resources)] for _ in
                     range(max(schedule, key=lambda x: x[1])[1] + max(schedule, key=lambda x: x[0].time)[0].time)]
        for act, start_time, resource_level in scheduled:
            for t in range(act.time):
                for r in range(act.resources):
                    temp[start_time + t][resource_level + r] += 1
        return temp

    def colision_point(resource_grid):
        for t in range(len(resource_grid)):
            for r in range(len(resource_grid[0])):
                if resource_grid[t][r] > 1:
                    return t
        return 0

    # Process each activity in the schedule
    scheduled = []
    for activity, start_time in schedule:
        while find_space_for_activity(activity.time, activity.resources, start_time) is None:
            item = colision_activity(scheduled, activity, start_time)
            scheduled.remove(item)
            scheduled.append((item[0], item[1], total_resources - item[0].resources))
            resource_grid = update_resource_grid(scheduled)
            point = colision_point(resource_grid)
            if point > 0:
                for act, start, level in scheduled:
                    if start == point:
                        scheduled.remove((act, start, level))
                        temp = find_space_for_activity(act.time, act.resources, start)
                        scheduled.append((act, start, temp))
            # draw_rectangles(scheduled, total_resources, total_time)
            resource_grid = update_resource_grid(scheduled)

        resource_level = find_space_for_activity(activity.time, activity.resources, start_time)
        scheduled.append((activity, start_time, resource_level))
        resource_grid = update_resource_grid(scheduled)
        # draw_rectangles(scheduled, total_resources, total_time)

    draw_rectangles(scheduled, total_resources, total_time)

def draw_rectangles(schedule, total_resource, total_time):
    """
    Draw multiple rectangles on the given axes.

    :param ax: Matplotlib axes object where rectangles will be drawn.
    :param rectangles_info: List of tuples, where each tuple contains:
                            (x, y, width, height, 'text')
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set the limits and labels of the axes
    ax.set_xlim(0, total_time + 2)
    ax.set_ylim(0, total_resource + 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Resources')
    ax.set_yticks(range(1, total_resource + 1))
    ax.set_xticks(range(1, total_time + 1))

    # Draw a horizontal line to represent the maximum resource level
    ax.hlines(total_resource, 0, total_time + 2, colors='black', linestyles='dotted', lw=2)

    for (act, start_time, start_resource) in schedule:
        # Draw the rectangle
        x = start_time
        y = start_resource
        width = act.time
        height = act.resources
        text = act.identifier
        rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        # Place the text in the center of the rectangle
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center')

    plt.show()

def uniform_crossover(father, mother):
    if not father or not mother or len(father) != len(mother):
        raise ValueError("Father and mother schedules must be non-empty and of equal length")

    len_schedule = len(father)
    q = random.randint(0, len_schedule - 2)

    daughter = []
    son = []

    random_sequence = [random.choice([0, 1]) for _ in range(len_schedule)]
    for number in random_sequence:
        if number == 0:
            for act in father:
                if act not in daughter:
                    daughter.append(act)
                    break
            for act in mother:
                if act not in son:
                    son.append(act)
                    break
        else:
            for act in mother:
                if act not in daughter:
                    daughter.append(act)
                    break
            for act in father:
                if act not in son:
                    son.append(act)
                    break

    # print(random_sequence)

    return son, daughter


def one_point_crossover(father, mother):
    if not father or not mother or len(father) != len(mother):
        raise ValueError("Father and mother schedules must be non-empty and of equal length")

    len_schedule = len(father)
    q = random.randint(0, len_schedule - 2)

    daughter = []
    son = []

    daughter.extend(mother[:q])
    son.extend(father[:q])

    for act in father:
        if act not in daughter and len(daughter) < len_schedule:
            daughter.append(act)

    for act in mother:
        if act not in son and len(son) < len_schedule:
            son.append(act)

    # print(q)
    return son, daughter

def two_point_crossover(father, mother):
    # Ensure father and mother schedules are not empty and have the same length
    if not father or not mother or len(father) != len(mother):
        raise ValueError("Father and mother schedules must be non-empty and of equal length")

    len_schedule = len(father)

    # Generating random crossover points q1 and q2
    q1 = random.randint(0, len_schedule - 2)
    q2 = random.randint(q1 + 1, len_schedule - 1)

    daughter = []
    son = []

    # Add first q1 elements from mother
    daughter.extend(mother[:q1])
    son.extend(father[:q1])

    # Add elements from father, checking from beginning to end
    for act in father:
        if act not in daughter and len(daughter) < q2:
            daughter.append(act)

    # Add remaining elements from mother, checking from beginning to end
    for act in mother:
        if act not in daughter and len(daughter) < len_schedule:
            daughter.append(act)
        if act not in son and len(son) < q2:
            son.append(act)

    for act in father:
        if act not in son and len(son) < len_schedule:
            son.append(act)

    # print(q1,q2)

    return son, daughter

def mutate_individual(individual, pmutation, predecessors):
    mutated_individual = individual.copy()

    for i in range(len(individual) - 1):
        # Check if mutation should occur
        if random.random() < pmutation:
            # Swap activities jIi and jIi+1 if it doesn't violate the precedence assumption
            if is_valid_swap(mutated_individual, i, i + 1, predecessors):
                mutated_individual[i], mutated_individual[i + 1] = mutated_individual[i + 1], mutated_individual[i]

    return mutated_individual

def is_valid_swap(individual, i, j, predecessors):
    if individual[i] in predecessors[individual[j]]:
        return False
    if individual[j] in individual[i].successors:
        return False
    return True

def ranking_selection(population):
    # Sort the population based on fitness (assuming fitness is the second element of the tuple)
    sorted_population = sorted(population, key=lambda x: x[4])

    # Cut half of the population
    half_population_size = len(sorted_population) // 2
    selected_population = sorted_population[:half_population_size]

    return selected_population

def evaluate_finish_time(schedule):
    max = 0
    for act, start_time in schedule:
        if max < start_time + act.time:
            max = start_time + act.time
    return max

def evaluate_fitness(individual, coefficient):
    return individual[1] + coefficient * individual[3]

def create_individual(activities, total_resource, coefficient, alternative_chains, sequence, sequence_pool):
    for item in sequence_pool:
        if sequence == item[0]:
            individual = item
            return individual
    individual = evaluate_sequence(activities, total_resource, coefficient, alternative_chains, sequence, sequence_pool)
    return individual

def replace_activity(activity_sequence, identifier, activity):
    temp = [activity if act.identifier == identifier else act for act in activity_sequence]
    return temp

def find_alternatives_for_activity(identifier, alternative_chains):
    alternatives = []
    for chain in alternative_chains:
        if identifier in chain:
            alternatives.extend(chain)
    # Remove duplicates and the original identifier
    return list(set(alternatives) - {identifier})

def create_alternative_sequences(activities, activity_sequence, alternative_chains):
    all_sequences = [activity_sequence]

    for act in activity_sequence:
        alternatives = find_alternatives_for_activity(act.identifier, alternative_chains)
        new_sequences = []
        for alt_id in alternatives:
            for sequence in all_sequences:
                new_act = next((a for a in activities if a.identifier == alt_id), None)
                if new_act:
                    new_sequences.append(replace_activity(sequence, act.identifier, new_act))
        all_sequences.extend(new_sequences)

    return all_sequences

def generate_full_enumeration(activities, alternative_chains):
    # Extract activities not in any alternative chain
    regular_activities = [act for act in activities if
                          not any(act.identifier in chain for chain in alternative_chains)]

    # Extract activities that are part of alternative chains
    alternative_activities = []
    for chain in alternative_chains:
        chain_activities = [act for act in activities if act.identifier in chain]
        alternative_activities.append(chain_activities)

    # Generate all combinations, taking one from each alternative chain
    all_combinations = []
    for combination in itertools.product(*alternative_activities):
        # Combine alternative chain activities with regular activities
        full_combination = list(combination) + regular_activities
        all_combinations.append(full_combination)

    return all_combinations

def evaluate_sequence(activities, total_resource, coefficient, alternative_chains, original_sequence, sequence_pool):
    sequences = create_alternative_sequences(activities, original_sequence, alternative_chains)
    first_sequence = sequences[0]
    first_schedule = create_schedule(first_sequence, total_resource)
    robust = []
    robust.append((first_sequence, evaluate_finish_time(first_schedule), 0))

    #Calculate slacks
    for sequence in sequences[1:]:
        slacks = 0
        schedule = create_schedule(sequence, total_resource)
        for i in range(0, len(schedule)):
            slacks += schedule[i][1] + schedule[i][0].time - first_schedule[i][1] - first_schedule[i][0].time
        robust.append((sequence, evaluate_finish_time(schedule), slacks))

    individuals = []
    #Calculate robustness
    for current_tuple in robust:
        # Calculate the sum of differences for the current tuple
        sum_of_differences = sum(t[2] - current_tuple[2] for t in robust)
        sum_of_differences /= len(robust)
        current_tuple = current_tuple + (sum_of_differences,)
        fitness = evaluate_fitness(current_tuple, coefficient)
        # Append this sum to the current tuple
        individuals.append(current_tuple + (fitness,))

    result = None
    for individual in individuals:
        if individual[0] == original_sequence:
            result = individual
        sequence_pool.append(individual)

    return result

def print_individual(individual):
    print_activity_sequence(individual[0])
    schedule = create_schedule(individual[0], total_resource)
    print_schedule_formatted(schedule)
    print("Makespan = ", individual[1], ", ",
          "Slacks = ", individual[2], ", ",
          "Robustness = ", individual[3], ", ",
          "Fitness = ", individual[4])
    print()
    # draw_schedule(schedule, total_resource)


def genetic_algorithm(activities, alternative_chains, instance, total_resource, sequence_pool):
    pop = []
    pop_length = 40
    gene_num = 25
    coefficient = 0.05
    predecessors = find_predecessors(instance)

    #Generate initial population
    for i in range(0, pop_length):
        sequence = generate_random_activity_sequence(instance)
        individual = create_individual(activities, total_resource, coefficient, alternative_chains, sequence, sequence_pool)
        pop.append(individual)

    #
    for i in range(0, gene_num):
        random.shuffle(pop)
        pairs = [pop[i:i+2] for i in range(0, len(pop), 2)]
        for pair in pairs:
            son_sequence, daughter_sequence = two_point_crossover(pair[0][0], pair[1][0])
            son_sequence = mutate_individual(son_sequence, 0.05, predecessors)
            daughter_sequence = mutate_individual(daughter_sequence, 0.05, predecessors)
            son_individual = create_individual(activities, total_resource, coefficient, alternative_chains, son_sequence, sequence_pool)
            daughter_individual = create_individual(activities, total_resource, coefficient, alternative_chains, daughter_sequence, sequence_pool)
            pop.append(son_individual)
            pop.append(daughter_individual)
        pop = ranking_selection(pop)

    # Print out the best solution
    for individual in pop[:1]:
        print_individual(individual)


def test_pool_x2():
    result = [0, 0, 0, 0]
    for i in range (0, 10):
        file_pattern = "project_instances/J15x2/*.csv"
        for file_path in glob.glob(file_pattern):
            compute_times = []
            total_resource, activities, alternative_chains = create_activities_from_csv(file_path)

            instances = generate_full_enumeration(activities, alternative_chains)
            sequence_pool = []

            # Test for all instances
            for instance in instances:
                start_time = time.time()
                genetic_algorithm(activities, alternative_chains, instance, total_resource, sequence_pool)
                end_time = time.time()
                compute_times.append(end_time - start_time)

            temp = compute_times[0]
            compute_times[0] = 1
            for j in range(1, len(compute_times)):
                compute_times[j] = compute_times[j] / temp

            result = [item1 + item2 for item1, item2 in zip(result, compute_times)]
        result = [x / 10 for x in result]
    print(result)

# Example usage
if __name__ == "__main__":

    # Test pool performance
    # test_pool_x2()


    # Test one file
    file_path = r"project_instances/J30x3/J30x3_11.csv"
    total_resource, activities, alternative_chains = create_activities_from_csv(file_path)
    instances = generate_full_enumeration(activities, alternative_chains)
    sequence_pool = []
    compute_times = []

    for instance in instances:
        start_time = time.time()
        genetic_algorithm(activities, alternative_chains, instance, total_resource, sequence_pool)
        end_time = time.time()
        compute_times.append(end_time - start_time)

    print(compute_times)

    # Test whole directory
    # result = []
    # file_pattern = "project_instances/J15x3/*.csv"
    # compute_times = []


    # for file_path in glob.glob(file_pattern):
    #     compute_times = []
    #     total_resource, activities, alternative_chains = create_activities_from_csv(file_path)
    #
    #     instances = generate_full_enumeration(activities, alternative_chains)
    #     sequence_pool = []
    #
    #     # Test for all instances
    #     for instance in instances:
    #         start_time = time.time()
    #         genetic_algorithm(activities, alternative_chains, instance, total_resource, sequence_pool)
    #         end_time = time.time()
    #         compute_times.append(end_time - start_time)
    #     print(compute_times)
    #     sequence_pool = sorted(sequence_pool, key=lambda x: x[4])
    #     best_individual = sequence_pool[0]
    #     print_individual(best_individual)
    #
    #     result.append((best_individual[1], best_individual[3], best_individual[4]))
    #
    # for item in result:
    #     print(item)

    # show compute time
    # print(compute_times)

    # Test for only one instance
    # instance_1 = instances[3]
    # # genetic_algorithm(activities, alternative_chains, instance_1, total_resource, sequence_pool)
    #
    # # Test a random activity list and its all alternative sequences
    # sequence = generate_random_activity_sequence(instance_1)
    #
    # # Test a specific activity list\
    # instance_1 = instances[0]
    # sequence = generate_activity_sequence(instance_1, (1,2,5,3,4,6,8,10,7,9,11))
    # 
    # individual = evaluate_sequence(activities, total_resource, 0.5, alternative_chains, sequence, sequence_pool)
    # 
    # for item in sequence_pool:
    #     print_activity_sequence(item[0])
    #     schedule = create_schedule(item[0], total_resource)
    #     draw_schedule(schedule, total_resource)
    #     print_schedule_formatted(schedule)
    #     for temp in item[1:]:
    #         print(temp, end=' ')
    #     print()

    # Create all possible alternative sequences
    # alternative_sequences = create_alternative_sequences(activities, original_sequence, alternative_chains)
    # alternative_schedules = []
    # for sequence in alternative_sequences:
    #     print_activity_sequence(sequence)
    #     schedule = create_schedule(sequence, total_resource)
    #     print_schedule_formatted(schedule)
    #     alternative_schedules.append((sequence, schedule))
    #     draw_schedule(schedule, total_resource)
    #

    # Test cross_over_function

    # instance_1 = instances[0]
    # father = generate_random_activity_sequence(instance_1)
    # mother = generate_random_activity_sequence(instance_1)
    #
    # print("father: ", end=' ')
    # print_activity_sequence(father)
    # print("mother: ", end=' ')
    # print_activity_sequence(mother)
    #
    # son, daughter = uniform_crossover(father, mother)
    #
    # print("son: ", end=' ')
    # print_activity_sequence(son)
    # son_schedule = create_schedule(son, 6)
    #
    # print("dau: ", end=' ')
    # print_activity_sequence(daughter)
    # daughter_schedule = create_schedule(daughter, 6)



    # Test mutation function
    # instance_1 = instances[0]
    # activity_sequence = generate_random_activity_sequence(instance_1)
    # print_activity_sequence(activity_sequence)
    # mutated = mutate_individual(activity_sequence, 0.5, find_predecessors(activities))
    # print_activity_sequence(mutated)
    # mutated_schedule = create_schedule(mutated, 6)
    # print_schedule_formatted(mutated_schedule)
    # draw_schedule(mutated_schedule, 6)

    # genetic_algorithm(instance_1, 6)
    # genetic_algorithm(instance_2, 6)
