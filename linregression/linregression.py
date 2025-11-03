
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1, 2, 3, 4, 5, 6])  # Hypothetical example: house sqft in 1,000s
y = np.array([6, 7, 10, 13, 13, 15])  # Hypothetical example: house price in 100,000s (target variable)

# Prepare data with bias term: x_n = [1, x]
X_with_bias = np.column_stack([np.ones(len(X)), X])

# Initialize weights: w = [w0, w1] (bias and slope)
w = np.array([0.0, 0.0])

# Hyperparameters
eta = 0.01  # learning rate η
max_iter = 50

# Store history
history = {}
steps_to_capture = [0, 5, 10, 20, 30, 49]
current_step = 0

# Training (for visualization only for now)
for k in range(max_iter):
    # Capture at the start of each epoch if needed
    if current_step in steps_to_capture:
        history[current_step] = (w[1], w[0])  # Store (slope, intercept)
    
    for i in range(len(X)):
        x_n = X_with_bias[i]  # [1, X[i]]
        Y_actual = y[i]
        
        # Compute predicted output
        Y_hat = np.dot(w, x_n)  # w^T · x_n
        
        # Compute scalar δ
        delta = Y_actual - Y_hat
        
        # Update weights
        w = w + eta * delta * x_n
    
    current_step += 1

if (max_iter - 1) in steps_to_capture and (max_iter - 1) not in history:
    history[max_iter - 1] = (w[1], w[0])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
steps_to_plot = [0, 5, 10, 20, 30, 49]

for idx, step in enumerate(steps_to_plot):
    if step in history:
        w_curr, b_curr = history[step]
        y_pred = w_curr * X + b_curr
        sse = np.sum((y - y_pred)**2)
        
        axes[idx].scatter(X, y, s=100, alpha=0.6)
        axes[idx].plot(X, y_pred, 'r-', linewidth=2)
        
        # Residuals
        for i in range(len(X)):
            axes[idx].plot([X[i], X[i]], [y[i], y_pred[i]], 
                          'gray', linestyle='--', alpha=0.5)
        
        axes[idx].set_title(f'Step {step}: SSE={sse:.2f}')
        axes[idx].grid(alpha=0.3)
    else:
        axes[idx].text(0.5, 0.5, f'Step {step}\nNot captured', 
                      ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f'Step {step}: No data')

plt.tight_layout()
plt.show()