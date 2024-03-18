import matplotlib.pyplot as plt

# Data for the first line
y_values_new = [45.85, 45.85, 45.8, 45.8, 45.8, 45.8, 45.8, 45.8, 45.8, 45.8,
                45.8, 45.8, 45.8, 44.85, 44.85, 44.85, 44.85, 44.85, 44.85, 44.85]
x_values_new = range(len(y_values_new))

# Data for the second line
y_values_second_line = [41.65] * 20  # 41.65 repeated 20 times
x_values_additional = range(len(y_values_second_line))

# Drawing the line chart
plt.figure(figsize=(10, 6))
plt.plot(x_values_new, y_values_new, marker='o', label="J15x2")
plt.plot(x_values_additional, y_values_second_line, label="J15x3")
plt.title("Fitness value & number of generations J15, POP = 10")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.xticks([0, 5, 10, 15, 20])  # Custom X-axis values
plt.grid(True)
plt.legend()
plt.show()
