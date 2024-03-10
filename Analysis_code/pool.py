import matplotlib.pyplot as plt

# Data
instance = [1]
extended_labels = ['Variant 1', 'Variant 2', 'Variant 3', 'Variant 4', 'Variant 5', 'Variant 6', 'Variant 7', 'Variant 8']

values = [1, 0.96788766, 0.936559585, 0.962026842]
additional_values_2 = [1, 1.006656879, 1.022024848, 0.950199037, 1.079847572, 0.991366328, 1.020860275, 1.062448673]
additional_values_3 = [1, 1.042305885, 1.080977906, 1.110426112]
additional_values_4 = [1, 1.05388455, 1.093594109, 1.140039467, 1.173934955, 1.22130413, 1.269328529, 1.290303504]


# Padding the original values and additional_values_3 to have 8 elements for consistency in plotting
padded_values = values + [None] * (len(extended_labels) - len(values))
padded_additional_values_3 = additional_values_3 + [None] * (len(extended_labels) - len(additional_values_3))

# Re-plotting with all the lines
plt.figure(figsize=(12, 6))
plt.plot(extended_labels, padded_values, marker='o', label='J15x2')
plt.plot(extended_labels, additional_values_2, marker='x', color='red', label='J15x3')
plt.plot(extended_labels, padded_additional_values_3, marker='^', color='green', label='J30x2')
plt.plot(extended_labels, additional_values_4, marker='s', color='purple', label='J30x3')

# Title and labels
plt.title('Line chart of compute time in percentage')
plt.xlabel('Variants')
plt.ylabel('Percent (100%)')
plt.grid(True)
plt.legend()

# Show plot
plt.show()