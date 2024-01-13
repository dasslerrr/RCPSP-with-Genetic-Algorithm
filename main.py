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

def print_schedule(schedule):
    for activity, start_time in schedule.items():
        print(f"Activity {activity.identifier} starts at time {start_time}")

def evaluate_schedule(schedule) -> int:
    return schedule[-1]

def print_schedule_formatted(schedule):
    formatted_output = ", ".join(f"({activity.identifier},{start_time})" for activity, start_time in schedule)
    print(formatted_output)

# Example usage
if __name__ == "__main__":
    # num_generations = 100
    # pop_size = 50
    # best_schedule, best_duration = genetic_algorithm(num_generations, pop_size)
    # print("Best schedule:", [activity for activity, _, _ in best_schedule])
    # print("Best duration:", best_duration)

    file_path = r"project_instances/instance1.csv"
    activities = create_activities_from_csv(file_path)

    sequence = generate_random_activity_sequence(activities)
    print_activity_sequence(sequence)

    predecessors = find_predecessors(activities)

    schedule = create_schedule(sequence, 4)
    print_schedule_formatted(schedule)





