import random

# Define total resources and number of activities
total_resources = 6
num_activities = 30

# Initialize activities dictionary, alternative chains list, and predecessors
activities = {}
new_alternative_chains = []
predecessors = {i: [] for i in range(2, num_activities + 1)}


# Function to generate successors
def generate_successors_v5(id, mandatory_successors, remaining_activities):
    successors = []

    if id == 1:  # First activity will have 2, 3, 4 as successors
        return [2, 3, 4]
    if id in [num_activities]:  # No successors for the last activity
        return []

    # If id + 1 is a remaining activity, add it as a successor
    if id + 1 in remaining_activities:
        successors.append(id + 1)
        remaining_activities.remove(id + 1)

    # Then, if there are still possible successors, choose one randomly
    if remaining_activities:
        # Prioritize choosing from remaining activities
        possible_successors = [activity for activity in remaining_activities if activity > id]
        if possible_successors:
            successor = random.choice(possible_successors)
            remaining_activities.remove(successor)
            successors.append(successor)
    elif mandatory_successors:
        # Then choose from the mandatory successors that have an ID greater than the current activity
        possible_successors = [succ for succ in mandatory_successors if succ > id]
        if possible_successors:
            successor = random.choice(possible_successors)
            mandatory_successors.remove(successor)
            successors.append(successor)

    # If no mandatory successors or remaining activities or none are greater than the current ID, select randomly
    if not successors:
        possible_successors = [aid for aid in range(id + 1, num_activities) if aid != id]
        if possible_successors:
            successors.append(random.choice(possible_successors))

    return successors


# Mandatory successors list: all activities except the first and last
mandatory_successors = list(range(2, num_activities - 1))
remaining_activities = list(range(5, num_activities - 1))


# Create initial activities with successors
activity_ids = list(range(1, num_activities + 1))
for id in activity_ids:
    resource = random.randint(1, total_resources - 1)
    time = random.randint(2, 7)

    successors = generate_successors_v5(id, mandatory_successors, remaining_activities)

    activities[id] = (resource, time, successors)

    for succ in successors:
        predecessors[succ].append(id)

if num_activities not in [succ for act in activities.values() for succ in act[2]]:
    # Choose a random activity from the later ones to be the predecessor of the largest ID
    random_predecessor = random.choice(range(num_activities - 1, num_activities))
    activities[random_predecessor][2].append(num_activities)
    predecessors[num_activities].append(random_predecessor)

# Number of alternative chains to create
num_alt_chains = 2

# Create duplicates for alternative chains
for original_id in random.sample(activity_ids, num_alt_chains):
    # Create a duplicate named like "2x"
    duplicate_name = f"'{original_id}x'"
    duplicate_resource = random.randint(1, total_resources)
    duplicate_time = random.randint(2, 7)
    activities[duplicate_name] = (duplicate_resource, duplicate_time, activities[original_id][2])

    # Add both original and duplicate to the alternative chains list
    new_alternative_chains.append((original_id, duplicate_name))

    # Update the predecessors' successor lists to include both the original and the duplicate
    for activity in activities.values():
        if original_id in activity[2]:  # If the original is a successor
            activity[2].append(duplicate_name)  # Add the duplicate as well

# Function to format successors correctly
def format_successors_updated(successors):
    if len(successors) >= 2:
        return f'"{{{",".join(map(str, successors))}}}"'
    elif successors:
        return f'{{{",".join(map(str, successors))}}}'
    else:
        return "{}"

# Prepare the CSV content
csv_content = f"total_resource,{total_resources},,\n---,,,\nalternative_chain,,,\n"
for index, chain in enumerate(new_alternative_chains, start=1):
    csv_content += f'{index},"({chain[0]},{chain[1]})",,\n'
csv_content += "---,,,\njobnr,resource,time,successors\n"
for id, (resource, time, successors) in sorted(activities.items(), key=lambda x: str(x[0])):
    successors_str = format_successors_updated(successors)
    csv_content += f"{id},{resource},{time},{successors_str}\n"

csv_content = csv_content.strip()  # Clean up trailing newline

# Output the CSV content
print(csv_content)
