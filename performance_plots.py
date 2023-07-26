import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import glob

folder = "Rwanda"
# folders could also be Uganda, Ethiopia
approach = "sixth"
test_prop = 0.2

with open(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/Saved data/performance_{approach}_test_size_{test_prop}_dropped.pkl', 'rb') as f:
    performance_df = pickle.load(f)

# ---- First performance plot ----
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 4))

# Plot accuracy, recall, and precision
ax.plot(performance_df['Classifier'], performance_df['Accuracy'], marker='o', label='Accuracy')
ax.plot(performance_df['Classifier'], performance_df['Recall'], marker='o', label='Recall')
ax.plot(performance_df['Classifier'], performance_df['Precision'], marker='o', label='Precision')

# Set plot labels and title
ax.set_xlabel('Methods')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison of Methods')

# Customize plot appearance
ax.legend()
ax.grid(True)

# Set the y-limits
ax.set_ylim(0.5, 0.9)  # Adjust the values as needed

# Save the figure with higher resolution (e.g., DPI = 300)
dpi = 300  # Adjust the DPI value as needed
plt.savefig(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/ML/Performance plots/performance_plot_{approach}_test_size_{test_prop}_dropped.png', dpi=dpi)

# Show the plot
plt.show()

# ----- First plot for recall ------
# List of test_prop values
test_props = [0.2, 0.3, 0.4, 0.5]
# Metric
metric = 'Accuracy'

fig, ax = plt.subplots(figsize=(8, 4))

# Iterate over test_prop values
for i, test_prop in enumerate(test_props):
    # Filter the dataframe based on the test_prop value
    with open(
            f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/Saved data/performance_{approach}_test_size_{test_prop}.pkl',
            'rb') as f:
        performance_df = pickle.load(f)

    # Define the pastel color for the line plot
    color = sns.color_palette('pastel')[i]

    # Plot the recall values with pastel color
    ax.plot(performance_df['Classifier'], performance_df[metric], marker='o', label=f'Test_prop={test_prop}', color=color)

# Set plot labels and title
ax.set_ylabel(f'{metric}')
ax.set_title(f'{metric} Comparison for Different Test Proportions')

# Customize plot appearance
ax.legend()
ax.grid(True)

# Set the y-limits
ax.set_ylim(0.55, 0.9)  # Adjust the values as needed

# Save the figure
plt.savefig(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/ML/Performance plots/{metric}_plot_{approach}.png')

# Show the plot
plt.show()





# ---- Seconds performance plot ----
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Define a color palette for the classifiers
colors = ['#8fbdd8', '#f7ae76', '#a8d8b9', '#f392bd', '#cbb0e3']

# Accuracy
plt.subplot(1, 3, 1)
sns.barplot(x='Accuracy', y='Classifier', data=performance_df, palette=colors)
plt.xlabel('Accuracy')
plt.ylabel('Classifier')
plt.xlim(0, 1)

# Recall
plt.subplot(1, 3, 2)
sns.barplot(x='Recall', y='Classifier', data=performance_df, palette=colors)
plt.xlabel('Recall')
plt.ylabel('')
plt.xlim(0, 1)

# Precision
plt.subplot(1, 3, 3)
sns.barplot(x='Precision', y='Classifier', data=performance_df, palette=colors)
plt.xlabel('Precision')
plt.ylabel('')
plt.xlim(0, 1)

plt.tight_layout()
# Save the figure
plt.savefig(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/ML/Performance plots/performance_plot2_{approach}_test_size_{test_prop}.png')

plt.show()


