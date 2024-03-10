import matplotlib.pyplot as plt

# Data for the plot
x_values = [0.05, 0.1, 0.25, 0.5]

# Fitness values for J30x2 and J15x2
fitness_J30x2 = [70.7375, 69.64083333, 59.78333333, 38.1125]
fitness_J15x2 = [43.50083333, 43.7225, 42.52083333, 37.30833333]

# Creating the line chart
plt.figure(figsize=(10, 6))
plt.plot(x_values, fitness_J30x2, label='J30x2', marker='o')
plt.plot(x_values, fitness_J15x2, label='J15x2', marker='s')

# Adding title and labels
plt.title('Fitness Value Comparison')
plt.xlabel('Coefficient')
plt.ylabel('Fitness Value')
plt.xticks(x_values)
plt.legend()

# Displaying the chart
plt.grid(True)
plt.show()
