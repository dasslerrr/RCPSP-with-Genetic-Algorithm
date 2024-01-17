import csv
import random
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
    resource = []

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

def evaluate_schedule(schedule) -> int:
    return schedule[-1]

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
    fig, ax = plt.subplots(figsize=(8, 5))

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

def two_point_crossover(father, mother):
    # Ensure father and mother schedules are not empty and have the same length
    if not father or not mother or len(father) != len(mother):
        raise ValueError("Father and mother schedules must be non-empty and of equal length")

    len_schedule = len(father)

    # Generating random crossover points q1 and q2
    q1 = random.randint(0, len_schedule - 2)
    q2 = random.randint(q1 + 1, len_schedule - 1)

    daughter = []

    # Add first q1 elements from mother
    daughter.extend(mother[:q1])

    # Add elements from father, checking from beginning to end
    for act in father:
        if act not in daughter and len(daughter) < q2:
            daughter.append(act)

    # Add remaining elements from mother, checking from beginning to end
    for act in mother:
        if act not in daughter and len(daughter) < len_schedule:
            daughter.append(act)

    print(q1,q2)

    return daughter

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

# Example usage
if __name__ == "__main__":
    # num_generations = 100
    # pop_size = 50
    # best_schedule, best_duration = genetic_algorithm(num_generations, pop_size)
    # print("Best schedule:", [activity for activity, _, _ in best_schedule])
    # print("Best duration:", best_duration)

    # Create activity from csv
    file_path = r"project_instances/instance3.csv"
    activities = create_activities_from_csv(file_path)

    # Test a random schedule
    # schedule = create_schedule(random_sequence, 6)
    # print_schedule_formatted(schedule)
    # draw_schedule(schedule, 6)


    # Test a particular activity list
    # sequence = generate_activity_sequence(activities, [1,2,5,4,7,3,6,9,8,10,11])
    # print_activity_sequence(sequence)
    # schedule = create_schedule(sequence, 6)
    # print_schedule_formatted(schedule)
    # draw_schedule(schedule, 6)

    # Test cross_over_function
    father = generate_random_activity_sequence(activities)
    mother = generate_random_activity_sequence(activities)

    print_activity_sequence(father)
    print_activity_sequence(mother)

    daughter = two_point_crossover(father, mother)
    print_activity_sequence(daughter)
    daughter_schedule = create_schedule(daughter, 6)
    print_schedule_formatted(daughter_schedule)
    draw_schedule(daughter_schedule, 6)

    # Test mutation function
    mutated = mutate_individual(daughter, 0.5, find_predecessors(activities))
    print_activity_sequence(mutated)
    mutated_schedule = create_schedule(mutated, 6)
    print_schedule_formatted(mutated_schedule)
    draw_schedule(mutated_schedule, 6)

