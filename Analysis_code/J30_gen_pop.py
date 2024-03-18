import matplotlib.pyplot as plt

# Data for the first line (original)
y_values_print = [79.85, 79.85, 79.85, 79.85, 79.85, 79.85, 79.8, 79.75, 79.75, 79.75,
                  79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75,
                  79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75, 79.75]
x_values_print = range(len(y_values_print))

# Data for the second line (new)
y_values_additional_line = [68.3, 68.3, 65.2375, 65.2, 65.2, 65.2, 65.1875, 65.1875, 65.025, 64.55,
                            64.55, 64.55, 64.3, 64.3, 63.9875, 63.9875, 63.9375, 63.9375, 63.6, 63.6,
                            63.6, 63.6, 63.6, 63.425, 63.425, 63.425, 63.425, 63.425, 63.425, 63.425]

# Subtracting 20 from each value in the first line
y_values_adjusted_first_line = [y - 25 for y in y_values_print]

# Subtracting 10 from each value in the second line
y_values_adjusted_line = [y - 10 for y in y_values_additional_line]

# Drawing the line chart with adjusted lines
plt.figure(figsize=(12, 6))
plt.plot(x_values_print, y_values_adjusted_first_line, marker='o', label="J30x2")
plt.plot(x_values_print, y_values_adjusted_line, label="J30x3")
plt.title("Fitness value & number of generations J30, POP = 30")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend()
plt.show()