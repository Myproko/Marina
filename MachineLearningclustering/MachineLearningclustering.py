import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
# Data points with their features
data = {
    'P1': [0, 1, 0],
    'P2': [1, 1, 0],
    'P3': [0, 0, 1],
    'P4': [0, 0, 0],
    'P7': [1, 0, 1],
    'P10': [0, 1, 1]
}

# Final cluster assignments after convergence
cluster_labels = {
    'P1': 'Sad',
    'P2': 'Sad',
    'P3': 'Happy',
    'P4': 'Sad',
    'P7': 'Happy',
    'P10': 'Sad'
}

# Convert to matrix for PCA
points = np.array(list(data.values()))
labels = list(data.keys())
cluster_colors = ['blue' if cluster_labels[label] == 'Happy' else 'red' for label in labels]

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
points_2d = pca.fit_transform(points)

# Plotting
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    plt.scatter(points_2d[i, 0], points_2d[i, 1], color=cluster_colors[i], label=label, s=100)
    plt.text(points_2d[i, 0] + 0.02, points_2d[i, 1], label, fontsize=12)

# Legend
happy_patch = plt.Line2D([0], [0], marker='o', color='w', label='Happy',
                         markerfacecolor='blue', markersize=10)
sad_patch = plt.Line2D([0], [0], marker='o', color='w', label='Sad',
                       markerfacecolor='red', markersize=10)
plt.legend(handles=[happy_patch, sad_patch])

plt.title('K-Means Clustering Visualization (After Convergence)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()
