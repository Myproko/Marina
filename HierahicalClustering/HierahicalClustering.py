import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Step 1: Load your CSV dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your actual file path

# Step 2: Select features for clustering
features = df[['feature1', 'feature2', 'feature3']]  # Replace with relevant column names

# Step 3: Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Compute linkage matrix
linked = linkage(scaled_features, method='ward')  # You can also use 'single', 'complete', 'average'

# Step 5: Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
#Notes
#Replace 'feature1', 'feature2', 'feature3' with the actual column names you want to cluster on.

#You can experiment with different linkage methods ('ward', 'single', 'complete', 'average') depending on your data structure.

#aset has categorical variables, you'll need to encode them before clustering.

#If you upload your .csv file here, I can help you run this directly or tailor the code to your dataset.
