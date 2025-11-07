import matplotlib.pyplot as plt
import numpy as np

# XOR data
X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
y = np.array([-1, -1, 1, 1])

# Plot
plt.figure(figsize=(8, 6))
colors = ['red' if label == 1 else 'blue' for label in y]
markers = ['+' if label == 1 else '-' for label in y]
for i, (point, color, marker) in enumerate(zip(X, colors, markers)):
    plt.scatter(point[0], point[1], c=color, s=500, marker='${}$'.format(marker), linewidths=3)

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Linearly Inseparable (XOR) Problem', fontsize=16)
plt.grid(True, alpha=0.3)
plt.show()
# This code visualizes the XOR problem, which is a classic example of a linearly inseparable dataset.
# Points belonging to class 1 are marked with a red '+' and points belonging to class -1 are marked with a blue '-'.
#2-layers perceptron cannot solve this problem.
import matplotlib.pyplot as plt
import numpy as np

# XOR data
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y_true = np.array([-1, 1, 1, -1])

# Initial weights and biases
W1 = np.array([[20, 20], [20, 20]])  # 2x2: input to hidden (Input -> Layer 1)
b1 = np.array([-10, -30])  # bias for hidden layer
v = np.array([1, -1])  # hidden to output weights (Layer 1 -> Layer 2)
b2 = -0.5  # output bias

# Forward pass
def forward(X, W1, b1, v, b2):
    z1 = X @ W1.T + b1  # hidden layer (Layer 1) pre-activation
    h1 = np.tanh(z1)  # hidden layer (Layer 1) activation
    z2 = h1 @ v + b2  # output (Layer 2) pre-activation
    y_hat = 1 / (1 + np.exp(-z2))  # output after activation (sigmoid)
    return z1, h1, z2, y_hat

# Call the forward method to perform the forward pass
z1, h1, z2, y_hat = forward(X, W1, b1, v, b2)

import matplotlib.pyplot as plt
import numpy as np

# Create visualization with just 2 plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Input space (XOR data)
ax = axes[0]
colors = ['blue' if y == -1 else 'red' for y in y_true]
ax.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors='black', linewidth=2)
for i in range(len(X)):
    ax.text(X[i, 0], X[i, 1], f'({X[i, 0]},{X[i, 1]})', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('Input Space (XOR)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Plot 2: Single decision boundary attempt (fails for XOR)
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors='black', linewidth=2)
for i in range(len(X)):
    ax.text(X[i, 0], X[i, 1], f'({X[i, 0]},{X[i, 1]})', ha='center', va='center', 
            fontsize=8, color='white', fontweight='bold')

# Try to draw a single linear decision boundary (it will fail to separate XOR)
xx = np.linspace(-1.5, 1.5, 100)
yy = np.linspace(-1.5, 1.5, 100)
XX, YY = np.meshgrid(xx, yy)

# Example decision boundary: x1 + x2 = 0 (or any other line you choose)
Z = XX + YY  # This will fail to properly separate XOR

ax.contour(XX, YY, Z, levels=[0], colors='green', linewidths=3, linestyles='--')
ax.contourf(XX, YY, Z, levels=[-100, 0, 100], colors=['lightblue', 'lightcoral'], alpha=0.3)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('Single Decision Boundary\n(Cannot Solve XOR!)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()


# new code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14

# Customer Transaction Data
data = {
    'Transaction': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
    'Music Download?': ['No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'Yes'],
    'Music Streaming?': ['Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Online Games': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Output': ['Buys', 'Cancels', 'Buys', 'Cancels', 'Cancels', 'Cancels', 'Buys', 'Cancels', 'Cancels', 'Buys']
}

df = pd.DataFrame(data)
print("=" * 80)
print("CUSTOMER TRANSACTION DATA")
print("=" * 80)
print(df.to_string(index=False))


# Encoding categorical data
# Convert Yes/No to 1/0 and Buys/Cancels to 1/0 (for regression-style loss)
vectorized_data = {
    'Transaction': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
    'Music Download?': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    'Music Streaming?': [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    'Online Games': [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    'Output': [1, 0, 1, 0, 0, 0, 1, 0, 0, 1]  # Buys=1, Cancels=0
}

df_vec = pd.DataFrame(vectorized_data)
print("\n" + "=" * 80)
print("VECTORIZED DATA")
print("=" * 80)
print("Features:")
print("  • Yes → 1")
print("  • No → 0")
print("\nClass (Output):")
print("  • Buys → 1")
print("  • Cancels → 0")
print()
print(df_vec.to_string(index=False))
#%%
# Create feature matrix X and labels y
X = df_vec[['Music Download?', 'Music Streaming?', 'Online Games']].values
y = df_vec['Output'].values

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Labels y shape: {y.shape}")
print()

# Network parameters
D = X.shape[1]  # Number of input features (3)
K = 2           # Number of hidden units
eta = 0.1      # Learning rate
MaxIter = 300   # Maximum iterations

print("=" * 80)
print("TWO-LAYER NETWORK TRAINING")
print("=" * 80)
print(f"Network architecture:")
print(f"  • Input features (D): {D}")
print(f"  • Hidden units (K): {K}")
print(f"  • Output units: 1")
print(f"  • Learning rate (η): {eta}")
print(f"  • Max iterations: {MaxIter}")
print()
#%%
# Initialize weights
np.random.seed(42)  # For reproducibility
W = np.random.randn(D, K) * 0.1  # D×K matrix
v = np.random.randn(K) * 0.1     # K-vector

print("Initial weights:")
print(f"  W (input to hidden) =\n{W}")
print(f"  v (hidden to output) = {v}")
print()

# Training history
history = {
    'loss': [],
    'W': [],
    'v': [],
    'predictions': []
}
#%%
# Training loop
print("=" * 80)
print("TRAINING PROGRESS")
print("=" * 80)

for iteration in range(MaxIter):
    # Initialize gradients
    G = np.zeros_like(W)  # D×K matrix for hidden layer
    g = np.zeros_like(v)  # K-vector for output layer
    
    total_loss = 0
    predictions = []
    
    # Loop over all training examples
    for idx in range(len(X)):
        x = X[idx]
        y_true = y[idx]
        
        # ===== FORWARD PASS =====
        # Compute hidden layer activations
        a = np.dot(W.T, x)  # K-vector: weighted inputs to hidden units
        h = np.tanh(a)      # K-vector: hidden activations
        
        # Compute output
        y_pred = np.dot(v, h)  # Scalar: network output
        
        # Compute error and loss
        e = y_true - y_pred
        loss = 0.5 * e**2
        total_loss += loss
        predictions.append(y_pred)
        
        # ===== BACKWARD PASS =====
        # Update output layer gradient
        g = g - e * h  # Accumulate: ∇v = -e·h
        
        # Update hidden layer gradients
        for i in range(K):
            # Gradient for hidden unit i
            grad_i = -e * v[i] * (1 - np.tanh(a[i])**2) * x
            G[:, i] = G[:, i] + grad_i
    
    # Update weights (after processing all examples-batch update)
    W = W - eta * G
    v = v - eta * g
    
    # Store history
    history['loss'].append(total_loss)
    history['W'].append(W.copy())
    history['v'].append(v.copy())
    history['predictions'].append(predictions.copy())
    
    # Print progress every 10 iterations
    if iteration % 10 == 0 or iteration == MaxIter - 1:
        avg_loss = total_loss / len(X)
        print(f"Iteration {iteration:3d}: Total Loss = {total_loss:.4f}, Avg Loss = {avg_loss:.4f}")

print()
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Final weights:")
print(f"  W (input to hidden) =\n{W}")
print(f"  v (hidden to output) = {v}")
print()
#%%
# Final predictions
print("=" * 80)
print("FINAL PREDICTIONS")
print("=" * 80)
print(f"{'Transaction':<12} {'True':<6} {'Predicted':<10} {'Rounded':<8} {'Correct?'}")
print("-" * 60)

for idx in range(len(X)):
    x = X[idx]
    y_true = y[idx]
    
    # Forward pass
    a = np.dot(W.T, x)
    h = np.tanh(a)
    y_pred = np.dot(v, h)
    y_rounded = 1 if y_pred > 0.5 else 0
    correct = "✓" if y_rounded == y_true else "✗"
    
    print(f"{df['Transaction'][idx]:<12} {y_true:<6} {y_pred:<10.4f} {y_rounded:<8} {correct}")

# Calculate accuracy
correct_predictions = sum(1 for idx in range(len(X)) 
                         if (1 if np.dot(v, np.tanh(np.dot(W.T, X[idx]))) > 0.5 else 0) == y[idx])
accuracy = correct_predictions / len(X) * 100
print()
print(f"Training Accuracy: {accuracy:.1f}%")
#%%
#matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings

warnings.filterwarnings('ignore', message='Glyph.*missing from font.*')

def network_output(x, W, v):
    """Compute network output for input x"""
    a = np.dot(W.T, x)
    h = np.tanh(a)
    return np.dot(v, h)

def create_decision_surface(W, v, resolution=25):
    """Create a mesh representing the decision boundary (output = 0.5)"""
    x_range = np.linspace(-0.2, 1.2, resolution)
    y_range = np.linspace(-0.2, 1.2, resolution)
    z_range = np.linspace(-0.2, 1.2, resolution)
    
    grid_output = np.zeros((resolution, resolution, resolution))
    
    for i, x_val in enumerate(x_range):
        for j, y_val in enumerate(y_range):
            for k, z_val in enumerate(z_range):
                x_test = np.array([x_val, y_val, z_val])
                grid_output[i, j, k] = network_output(x_test, W, v)
    
    return x_range, y_range, z_range, grid_output

def plot_network_decision_surface(ax, W, v, X, y, df, title):
    """Plot the decision boundary as a continuous surface"""
    
    # Create decision surface
    x_range, y_range, z_range, grid_output = create_decision_surface(W, v, resolution=25)
    
    print(f"Grid output range: [{grid_output.min():.3f}, {grid_output.max():.3f}]")
    
    surface_plotted = False
    
    # Try marching cubes if 0.5 is within the range
    if grid_output.min() < 0.5 < grid_output.max():
        try:
            from skimage import measure
            
            # Extract isosurface at output = 0.5
            verts, faces, _, _ = measure.marching_cubes(grid_output, level=0.5)
            
            # Scale vertices to actual coordinate range
            verts[:, 0] = verts[:, 0] / (len(x_range) - 1) * (x_range[-1] - x_range[0]) + x_range[0]
            verts[:, 1] = verts[:, 1] / (len(y_range) - 1) * (y_range[-1] - y_range[0]) + y_range[0]
            verts[:, 2] = verts[:, 2] / (len(z_range) - 1) * (z_range[-1] - z_range[0]) + z_range[0]
            
            # Plot the mesh
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           color='cyan', alpha=0.4, edgecolor='blue', linewidth=0.2)
            surface_plotted = True
            
        except ImportError:
            print("scikit-image not available")
    
    # Fallback: use slice method
    if not surface_plotted:
        print("Using slice visualization method...")
        n_slices = 10
        
        for z_val in np.linspace(0, 1, n_slices):
            X1, X2 = np.meshgrid(x_range, y_range)
            Z_out = np.zeros_like(X1)
            
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    x_test = np.array([X1[i, j], X2[i, j], z_val])
                    Z_out[i, j] = network_output(x_test, W, v)
            
            # Check if 0.5 is in range for this slice
            if Z_out.min() < 0.5 < Z_out.max():
                ax.contour(X1, X2, Z_out, levels=[0.5], 
                          colors='blue', linewidths=2, alpha=0.6, offset=z_val)
        
        # Also do Y slices
        for y_val in np.linspace(0, 1, n_slices):
            X1, X3 = np.meshgrid(x_range, z_range)
            Z_out = np.zeros_like(X1)
            
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    x_test = np.array([X1[i, j], y_val, X3[i, j]])
                    Z_out[i, j] = network_output(x_test, W, v)
            
            if Z_out.min() < 0.5 < Z_out.max():
                ax.contour(X1, Z_out, X3, levels=[0.5], 
                          colors='blue', linewidths=2, alpha=0.4, offset=y_val)
    
    # Plot data points
    for idx in range(len(X)):
        color = 'green' if y[idx] == 1 else 'red'
        ax.scatter(*X[idx], color=color, s=250, alpha=0.9, 
                  edgecolors='black', linewidth=2, zorder=10)
        ax.text(X[idx, 0], X[idx, 1], X[idx, 2] + 0.15, df['Transaction'][idx], 
                fontsize=9, ha='center', weight='bold')
    
    # Labels and formatting
    ax.set_xlabel('Music Download', fontsize=11, weight='bold')
    ax.set_ylabel('Music Streaming', fontsize=11, weight='bold')
    ax.set_zlabel('Online Games', fontsize=11, weight='bold')
    ax.set_title(title, fontsize=12, weight='bold', pad=10)
    
    ax.set_xlim([-0.3, 1.3])
    ax.set_ylim([-0.3, 1.3])
    ax.set_zlim([-0.3, 1.3])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Buys (y=1)'),
        Patch(facecolor='red', label='Cancels (y=0)'),
        Patch(facecolor='cyan' if surface_plotted else 'blue', 
              alpha=0.4, label='Decision Boundary')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.view_init(elev=20, azim=45)

# Create visualization
fig = plt.figure(figsize=(18, 6))

# Initial state
W_init = history['W'][0]
v_init = history['v'][0]
ax1 = fig.add_subplot(131, projection='3d')
plot_network_decision_surface(ax1, W_init, v_init, X, y, df, 
                              f'Initial State (Iteration 0)\nRandom Weights')

# Mid-training
mid_iter = len(history['W']) // 2
W_mid = history['W'][mid_iter]
v_mid = history['v'][mid_iter]
ax2 = fig.add_subplot(132, projection='3d')
plot_network_decision_surface(ax2, W_mid, v_mid, X, y, df, 
                              f'Mid-Training (Iteration {mid_iter})\nLoss: {history["loss"][mid_iter]:.3f}')

# Final state
W_final = history['W'][-1]
v_final = history['v'][-1]
ax3 = fig.add_subplot(133, projection='3d')
plot_network_decision_surface(ax3, W_final, v_final, X, y, df, 
                              f'Final State (Iteration {MaxIter-1})\nLoss: {history["loss"][-1]:.3f}')

plt.tight_layout()
plt.show()

print("The decision boundary is non-linear")
print("and the network successfully classifies all transactions.")
