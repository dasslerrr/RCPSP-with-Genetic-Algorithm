import matplotlib.pyplot as plt
import seaborn as sns

# Provided data
data = {
    'J15x2': [1.56512928, 2.29502964, 0.300030708, 2.47612381, 2.779976606, 1.138994455, 1.911068439, 1.052999973, 2.549026966, 1.899966002],
    'J15x3': [6.986454248, 1.064080238, 6.372730255, 2.80302906, 5.160223246, 7.282167196, 7.519160032, 8.18419981, 9.615687609, 10.83381319],
    'J30x2': [27.76893735, 24.20384383, 24.3040657, 27.38978219, 24.49518156, 26.79957747, 29.25688624, 21.52328801, 28.84486699, 26.71962905],
    'J30x3': [127.2622445, 126.6011508, 122.7921638, 103.5668654, 117.6270468, 128.7467353, 133.2170653, 126.6020298, 115.6928723, 135.1631308]
}

# Create a DataFrame for easier plotting
import pandas as pd
df = pd.DataFrame(data)

# Plotting J15x2 and J15x3 on the same plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['J15x2', 'J15x3']])
plt.title('Boxplot of J15x2 and J15x3')
plt.ylabel('Compute Time (seconds)')
plt.xlabel('Instance')
plt.grid(True)
plt.show()

# Plotting J30x2 and J30x3 on a different plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['J30x2', 'J30x3']])
plt.title('Boxplot of J30x2 and J30x3')
plt.ylabel('Compute Time (seconds)')
plt.xlabel('Instance')
plt.grid(True)
plt.show()