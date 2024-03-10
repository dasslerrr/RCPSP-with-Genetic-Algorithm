import matplotlib.pyplot as plt

# Data for the new line chart
x_values = [0.05, 0.1, 0.25, 0.5]
y_values_line1 = [43.03333333, 43.8, 46.46666667, 49.66666667]
y_values_line2 = [9.35, -0.775, -15.78333333, -24.71666667]
y_values_line3 = [70.5, 73.36666667, 79.7, 85.73333333]
y_values_line4 = [4.75, -37.25833333, -79.66666667, -95.24166667]
# Creating a line chart with two different y-axes: one for Line 1 and Line 3, and the other for Line 2 and Line 4

# Adjusting the provided code to include a legend for all four lines

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plotting Line 1 and Line 3 on the first y-axis
line1, = ax1.plot(x_values, y_values_line1, color='blue', linestyle='-', label='J15x2 Makespan')
line3, = ax1.plot(x_values, y_values_line3, marker='x', color='blue', linestyle='--', label='J30x2 Makespan')

# Setting labels and titles for the first y-axis
ax1.set_xlabel('Coefficient')
ax1.set_ylabel('Makespan', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# Creating a second y-axis
ax2 = ax1.twinx()

# Plotting Line 2 and Line 4 on the second y-axis
line2, = ax2.plot(x_values, y_values_line2, color='orange', linestyle='-', label='J15x2 Robustness')
line4, = ax2.plot(x_values, y_values_line4, marker='x', color='red', linestyle='--', label='J30x2 Robustness')

# Setting labels for the second y-axis
ax2.set_ylabel('Robustness', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Adding a title
plt.title('Line chart of makespan - robustness from J15x2 & J30x2')

# Adding a legend
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
plt.legend(lines, labels)

# Show plot
plt.show()
