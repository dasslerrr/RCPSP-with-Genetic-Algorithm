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

    for activity in activity_sequence:
        finish_time.sort()
        for current_time in finish_time:
            if is_precedence_feasible(scheduled, current_time, activity, predecessors):
                if is_resource_feasible(scheduled, current_time, activity, total_resources):
                    scheduled.append((activity, current_time))
                    finish_time.append(current_time + activity.time)
                    break

    return scheduled

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
    # Initialize the 2D array for resource tracking
    resource_grid = [[0 for _ in range(total_resources)] for _ in
                     range(max(schedule, key=lambda x: x[1])[1] + max(schedule, key=lambda x: x[0].time)[0].time)]

    total_time = schedule[-1][1] + schedule[-1][0].time
    # Function to find space for an activity
    def find_space_for_activity(activity_duration, activity_resources, schedule_start_time):
        for start_time in range(schedule_start_time, len(resource_grid) - activity_duration + 1):
            for resource_level in range(total_resources - activity_resources + 1):
                if all(resource_grid[start_time + t][resource_level + r] == 0 for t in range(activity_duration) for r in
                       range(activity_resources)):
                    return start_time, resource_level
        return None, None

    # Function to update the resource grid
    def update_resource_grid(start_time, resource_level, duration, resources):
        for t in range(duration):
            for r in range(resources):
                resource_grid[start_time + t][resource_level + r] = 1

    # Process each activity in the schedule
    rectangles_info = []
    for activity, start_time in schedule:
        start_time, resource_level = find_space_for_activity(activity.time, activity.resources, start_time)
        if start_time is not None and resource_level is not None:
            rectangles_info.append((start_time, resource_level, activity.time, activity.resources, activity.identifier))
            update_resource_grid(start_time, resource_level, activity.time, activity.resources)
            draw_rectangles(rectangles_info, total_resources, total_time)
    return rectangles_info

def draw_rectangles(rectangles_info, total_resource, total_time):
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

    for (x, y, width, height, text) in rectangles_info:
        # Draw the rectangle
        rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        # Place the text in the center of the rectangle
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center')

    plt.show()


# Example usage
if __name__ == "__main__":
    # num_generations = 100
    # pop_size = 50
    # best_schedule, best_duration = genetic_algorithm(num_generations, pop_size)
    # print("Best schedule:", [activity for activity, _, _ in best_schedule])
    # print("Best duration:", best_duration)

    file_path = r"project_instances/instance3.csv"
    activities = create_activities_from_csv(file_path)

    random_sequence = generate_random_activity_sequence(activities)
    print_activity_sequence(random_sequence)

    schedule = create_schedule(random_sequence, 6)
    print_schedule_formatted(schedule)
    draw_schedule(schedule, 6)

    # sequence = generate_activity_sequence(activities, [1,2,4,5,3,6,9,7,8,10,11])
    # print_activity_sequence(sequence)
    # schedule = create_schedule(sequence, 6)
    # print_schedule_formatted(schedule)


