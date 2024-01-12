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