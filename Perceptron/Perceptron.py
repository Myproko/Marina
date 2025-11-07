from IPython.display import IFrame, display

# Display the HTML file in an iframe
display(IFrame(src='hyperplane_1d.html', width='100%', height='1000'))
#%%
from IPython.display import IFrame, display

# Display the HTML file in an iframe
display(IFrame(src='hyperplane_2d.html', width='100%', height='1000'))
import numpy as np
import matplotlib.pyplot as plt

def create_hyperplane_plot():
    w = np.array([1, 1])  # weight vector-changing it will change the slope of the hyperplane
    b = 0  # bias - shifts the line away from origin
    
    # Example data points
    # In 2D, each point is just [x1, x2] coordinates
    data_points = np.array([[1.5, 0.5], [-0.5, -1.5], [-1, 1], [0.5, -0.5]])
    
    # The hyperplane is all points where w·x + b = 0
    # For our w=[1,1] and b=0, this means: 1*x1 + 1*x2 + 0 = 0, or simply: x1 + x2 = 0
    x1 = np.linspace(-2, 2, 100) # np.linspace(-2, 2, 100) creates 100 evenly spaced x1 values from -2 to 2
    # Just to make the example work, we calculate x2 to make wx+b = 0
    # In real-world ML problems, we should have x1 and x2 (and more) data points, and the model should calculate w (and b)
    x2 = -(w[0] * x1 + b) / w[1]  
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the decision boundary
    ax.plot(x1, x2, 'g-', linewidth=2, label='Decision boundary: x1 + x2 = 0')
    
    # Plot and classify the example points
    for point in data_points:
        activation = w[0]*point[0] + w[1]*point[1] + b  # This is w·x + b
        if activation > 0:
            ax.plot(point[0], point[1], 'b+', markersize=12, markeredgewidth=2)
            ax.text(point[0]+0.1, point[1]+0.1, f'{activation:.1f}', fontsize=9, color='blue')
        else:
            ax.plot(point[0], point[1], 'r_', markersize=12, markeredgewidth=2)
            ax.text(point[0]+0.1, point[1]+0.1, f'{activation:.1f}', fontsize=9, color='red')
    
    # Draw the weight vector w pointing positive side
    ax.arrow(0, 0, w[0], w[1], head_width=0.1, head_length=0.1, 
              fc='purple', ec='purple', linewidth=2, label='w vector (perpendicular)')
    
    # Add annotations
    ax.text(0.3, 0.8, 'POSITIVE SIDE\n(w·x + b > 0)', fontsize=11, color='blue', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    ax.text(-1.5, -0.8, 'NEGATIVE SIDE\n(w·x + b < 0)', fontsize=11, color='red',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink", alpha=0.5))
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('The weight vector w defines the hyperplane and points toward positive classifications')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    return fig, w, b
#%%
fig, w, b = create_hyperplane_plot()
plt.show()

# Quick explanation
print("What's happening here:")
print(f"• The line is where w·x + b = 0, which is: {w[0]}*x1 + {w[1]}*x2 + {b} = 0")
print(f"• Points with w·x + b > 0 are classified as positive (+)")
print(f"• Points with w·x + b < 0 are classified as negative (-)")
print(f"• The weight vector w is perpendicular to the boundary and points to the + side")
print(f"• Training a perceptron (or a full model later on) is all about learning what W and b should be, i.e. the decision boundary")
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Showing the image example

img = Image.open('images/neuron.png')
plt.imshow(img)
plt.axis('off')
plt.show()

# Input vector
x = np.array([1, 2, -1, 2])
print("Input vector x =", x)

# Weights from the slide  
w = np.array([3, -3, 1, 3])
print("Weights w =", w)

# Compute activation function: a = Σ(wi * xi)
a = np.dot(w, x)
print("\nActivation calculation: a = Σ(wi·xi)")
print(f"a = w·x = {w[0]}·{x[0]} + {w[1]}·{x[1]} + {w[2]}·{x[2]} + {w[3]}·{x[3]}")
print(f"a = {w[0]*x[0]} + {w[1]*x[1]} + {w[2]*x[2]} + {w[3]*x[3]}")
print(f"a = {a}")

# Apply threshold function
if a > 0:
    output = +1
    print(f"\nSince a = {a} > 0 → output = +1 (positive example)")
else:
    output = -1
    print(f"\nSince a = {a} ≤ 0 → output = -1 (negative example)")
    import numpy as np

# Training data (x, y) where x = features, y = labels
data = [
    (np.array([1, 0]), +1),    # positive example
    (np.array([0, 1]), -1),    # negative example  
    (np.array([1, 1]), +1),    # positive example
    (np.array([-1, 0]), -1),   # negative example
]

# Algorithm: PerceptronTrain(data, maxIter)
maxIter = 3  # number of learning iterations
D = 2  # dimension

# Store history for visualization
history = {'w': [], 'b': [], 'iter': [], 'example': []}

# 1. Initialize weights
w = np.zeros(D)  # initialize weight vector with 0s
print(f"1: Initialize weights w = {w}")

# 2. Initialize bias
b = 0
print(f"2: Initialize bias b = {b}")

# Store initial state for visualization
history['w'].append(w.copy())
history['b'].append(b)
history['iter'].append(0)
history['example'].append(0)

# 3. For iter = 1 ... maxIter do
for iter in range(1, maxIter + 1):   # for each iteration in maxIter
    print(f"\n3: Iteration {iter}")
    print("=" * 30)
    
    # 4. For all (x, y) in data do
    for i, (x, y) in enumerate(data):  # for each example in the data
        print(f"  4: Example {i+1}: x={x}, y={y}")
        
        # 5. Compute activation
        a = np.dot(w, x) + b
        print(f"  5: a = w·x + b = {w}·{x} + {b} = {a}")
        
        # 6. If ya ≤ 0 then (misclassified)
        if y * a <= 0:
            print(f"  6: Misclassified! (y·a = {y*a} ≤ 0)")
            
            # 7. Update weights (learning!)
            w = w + y * x  # if error, w is increased by yx
            print(f"  7: w ← w + y·x = {w - y*x} + {y}·{x} = {w}")
            
            # 7. Update bias
            b = b + y  # b is increased by y
            print(f"  8: b ← b + y = {b - y} + {y} = {b}")

            # Store state after update for visualization
            history['w'].append(w.copy())
            history['b'].append(b)
            history['iter'].append(iter)
            history['example'].append(i+1)
            
            print(f"We moved activation function one step to the right direction")
        else:
            print(f"  6: Correct (y·a = {y*a} > 0)")  # no learning from this example if it was predicted correctly
        print("-" * 20)

# 12. Final weights
print(f"\n12: Final weights w = {w}, bias b = {b}")

# Visualization
n_states = len(history['w'])
fig, axes = plt.subplots(1, n_states, figsize=(4*n_states, 4))

if n_states == 1:
    axes = [axes]

for idx, ax in enumerate(axes):
    w_current = history['w'][idx]
    b_current = history['b'][idx]
    
    # Plot data points
    for x, y in data:
        color = 'blue' if y == 1 else 'red'
        marker = 'o' if y == 1 else 's'
        ax.scatter(x[0], x[1], c=color, s=100, marker=marker, 
                  edgecolors='black', linewidth=1.5, alpha=0.7)
    
    # Plot decision boundary if weights are not zero
    if np.any(w_current != 0):
        # Create grid for visualization
        xlim = [-1.5, 1.5]
        ylim = [-0.5, 1.5]
        
        # Decision boundary: w1*x1 + w2*x2 + b = 0
        # Solve for x2: x2 = -(w1*x1 + b) / w2
        if w_current[1] != 0:  # Avoid division by zero
            x1_line = np.linspace(xlim[0], xlim[1], 100)
            x2_line = -(w_current[0] * x1_line + b_current) / w_current[1]
            ax.plot(x1_line, x2_line, 'g-', linewidth=2, label='Decision boundary')
        elif w_current[0] != 0:  # Vertical line case
            x1_boundary = -b_current / w_current[0]
            ax.axvline(x=x1_boundary, color='g', linewidth=2, label='Decision boundary')
    
    # Formatting
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    
    # Title
    if idx == 0:
        ax.set_title(f'Initial State\nw={w_current}, b={b_current}')
    else:
        ax.set_title(f'After Iter {history["iter"][idx]}, Example {history["example"][idx]}\n'
                    f'w={w_current}, b={b_current}')
    
    # Add legend
    ax.plot([], [], 'bo', label='Positive (+1)', markersize=8)
    ax.plot([], [], 'rs', label='Negative (-1)', markersize=8)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.suptitle('Perceptron Learning: Decision Boundary Evolution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Define the bias range
b_range = np.linspace(-2e-7, 2e-7, 1000)

# Compute 0-1 loss
# Assume w=[0,0] and x=[1,0], then we have a = b (activation equals bias)
# For y=+1, loss = 1 if a ≤ 0, loss = 0 if a > 0
losses = []
for b in b_range:
    a = b  # activation = bias (since w·x = 0)
    loss = 1 if a <= 0 else 0  # 0-1 loss based on sign of activation
    losses.append(loss)

# Plot the 0-1 loss function
ax.plot(b_range * 1e7, losses, 'k-', linewidth=3)

# Fill regions to show loss areas
ax.fill_between(b_range * 1e7, 0, losses, where=(np.array(b_range) <= 0), 
                alpha=0.15, color='red')
ax.fill_between(b_range * 1e7, 0, losses, where=(np.array(b_range) > 0), 
                alpha=0.15, color='blue')

# Mark the decision boundary
ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
ax.text(0, 1.15, 'Decision\nBoundary', ha='center', fontsize=10, fontweight='bold')

# Add "Misclassified" and "Correct" labels
ax.text(-1.3, 0.85, 'Misclassified\n(a ≤ 0)', fontsize=12, fontweight='bold', color='red')
ax.text(1.0, 0.05, 'Correct\n(a > 0)', fontsize=12, fontweight='bold', color='blue')

# Initial point: b = -0.0000001 = -1×10⁻⁷
b_initial = -1e-7
ax.scatter([-1], [1], color='red', s=250, zorder=5, 
           edgecolor='black', linewidth=2)
ax.text(-1, 1.08, 'Initial\nb = -0.0000001', fontsize=11, fontweight='bold', 
         color='red', ha='center')

# After update 1: b = b + 0.00000011 = -1e-7 + 1.1e-7 = 0.1e-7
b_update1 = b_initial + 1.1e-7
ax.scatter([0.1], [0], color='green', s=250, zorder=5,
           edgecolor='black', linewidth=2)
ax.arrow(-0.08, 0.15, 0.15, 0, head_width=0.04, head_length=0.05, 
         fc='green', ec='green', lw=2, alpha=0.8)
ax.text(0.1, 0.19, 'b + 0.00000011', fontsize=10, color='green', fontweight='bold', ha='center')
ax.text(0.1, -0.08, '→ Loss = 0', fontsize=11, color='green', fontweight='bold', ha='center')

# After update 2: b = b + 0.00000009 = -1e-7 + 0.9e-7 = -0.1e-7
b_update2 = b_initial + 0.9e-7
ax.scatter([-0.1], [1], color='orange', s=250, zorder=5,
           edgecolor='black', linewidth=2)
ax.arrow(-0.55, 0.85, 0.42, 0, head_width=0.04, head_length=0.05, 
         fc='orange', ec='orange', lw=2, alpha=0.8)
ax.text(-0.3, 0.89, 'b + 0.00000009', fontsize=10, color='orange', fontweight='bold', ha='center')
ax.text(-0.1, 0.92, '→ Loss = 1', fontsize=11, color='orange', fontweight='bold', ha='center')

ax.set_xlabel('Bias b (×10⁻⁷)', fontsize=13)
ax.set_ylabel('0-1 Loss', fontsize=13)
ax.set_title('The 0-1 Loss Problem', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([-2, 2])
ax.set_ylim([-0.1, 1.25])
ax.set_xticks([-2, -1, 0, 1, 2])

plt.tight_layout()
plt.show()

# Print the calculation
print("=" * 60)
print("THE 0-1 LOSS PROBLEM")
print("=" * 60)
print("\nInitial state:")
print(f"  b = -0.0000001 = -1×10⁻⁷")
print(f"  a = -1×10⁻⁷ (negative) → Loss = 1 (misclassified)")

print("\nResults of the two updates from the initial state:")
print("\n1. Update b ← b + 0.00000011:")
print(f"   New b = -1×10⁻⁷ + 1.1×10⁻⁷ = 0.1×10⁻⁷ (positive!)")
print(f"   → Loss = 0 (correct!)")

print("\n2. Update b ← b + 0.00000009:")
print(f"   New b = -1×10⁻⁷ + 0.9×10⁻⁷ = -0.1×10⁻⁷ (negative)")
print(f"   → Loss = 1 (misclassified)")

print("\nThe difference between updates is only 0.00000002,")
print("but one crosses the decision boundary and the other doesn't!")
print("\nThis is why 0-1 loss is non-smooth and has no efficient optimum solution.")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('How the Second Derivative Creates Convex vs Concave Shapes', fontsize=16, fontweight='bold')

# Setup x values
x = np.linspace(-3, 3, 200)

# Define example functions
convex_func = x**2
concave_func = -x**2 + 9

# Initialize plots
# Top left: Convex function
ax1.set_xlim(-3, 3)
ax1.set_ylim(-1, 10)
ax1.set_title('CONVEX: f(x) = x²', fontweight='bold', color='darkblue')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot of the function
ax1.plot(x, convex_func, 'b-', linewidth=2, alpha=0.3)

# Elements to animate
point_convex, = ax1.plot([], [], 'ro', markersize=10, label='Current point')
tangent_convex, = ax1.plot([], [], 'g-', linewidth=2, label='Tangent (slope)')
slope_arrow_convex, = ax1.plot([], [], 'g-', linewidth=3)
slope_text_convex = ax1.text(0, 8, '', fontsize=12, ha='center')
curvature_text_convex = ax1.text(0, 7, '', fontsize=11, ha='center', color='orange')

# Top right: Concave function
ax2.set_xlim(-3, 3)
ax2.set_ylim(-1, 10)
ax2.set_title('CONCAVE: f(x) = -x² + 9', fontweight='bold', color='darkblue')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# Plot of the function
ax2.plot(x, concave_func, 'b-', linewidth=2, alpha=0.3)

# Elements to animate
point_concave, = ax2.plot([], [], 'ro', markersize=10, label='Current point')
tangent_concave, = ax2.plot([], [], 'g-', linewidth=2, label='Tangent (slope)')
slope_arrow_concave, = ax2.plot([], [], 'g-', linewidth=3)
slope_text_concave = ax2.text(0, 8, '', fontsize=12, ha='center')
curvature_text_concave = ax2.text(0, 7, '', fontsize=11, ha='center', color='purple')

# Bottom left: Slope progression for convex
ax3.set_xlim(-3, 3)
ax3.set_ylim(-7, 7)
ax3.set_title("Slope (f') Over Time - INCREASING", fontweight='bold', color='green')
ax3.set_xlabel('Position (x)')
ax3.set_ylabel('Slope value')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=1)
ax3.axvline(x=0, color='k', linewidth=0.5)

# Track slope history
slope_history_x_convex = []
slope_history_y_convex = []
slope_line_convex, = ax3.plot([], [], 'g-', linewidth=2, alpha=0.7)
slope_point_convex, = ax3.plot([], [], 'go', markersize=8)

# Bottom right: Slope progression for concave
ax4.set_xlim(-3, 3)
ax4.set_ylim(-7, 7)
ax4.set_title("Slope (f') Over Time - DECREASING", fontweight='bold', color='green')
ax4.set_xlabel('Position (x)')
ax4.set_ylabel('Slope value')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=1)
ax4.axvline(x=0, color='k', linewidth=0.5)

# Track slope history
slope_history_x_concave = []
slope_history_y_concave = []
slope_line_concave, = ax4.plot([], [], 'g-', linewidth=2, alpha=0.7)
slope_point_concave, = ax4.plot([], [], 'go', markersize=8)

# Add legend
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

# Animation parameters
n_frames = 100
x_positions = np.linspace(-2.5, 2.5, n_frames)

def animate(frame):
    # Current x position
    x_pos = x_positions[frame]
    
    # CONVEX FUNCTION
    y_convex = x_pos**2
    slope_convex = 2 * x_pos  # f'(x) = 2x
    second_deriv_convex = 2  # f''(x) = 2
    
    # Update point
    point_convex.set_data([x_pos], [y_convex])
    
    # Update tangent line
    x_tangent = np.linspace(x_pos - 1, x_pos + 1, 50)
    y_tangent = y_convex + slope_convex * (x_tangent - x_pos)
    tangent_convex.set_data(x_tangent, y_tangent)
    
    # Add slope direction arrow
    arrow_scale = 0.3
    if abs(slope_convex) > 0.1:
        arrow_x = [x_pos, x_pos + arrow_scale]
        arrow_y = [y_convex, y_convex + arrow_scale * slope_convex]
        slope_arrow_convex.set_data(arrow_x, arrow_y)
        slope_arrow_convex.set_marker('>')
        slope_arrow_convex.set_markersize(10)
    
    # Update text
    slope_text_convex.set_text(f'Slope = {slope_convex:.1f}')
    slope_text_convex.set_position((x_pos, y_convex + 2))
    
    # Update slope history
    slope_history_x_convex.append(x_pos)
    slope_history_y_convex.append(slope_convex)
    slope_line_convex.set_data(slope_history_x_convex, slope_history_y_convex)
    slope_point_convex.set_data([x_pos], [slope_convex])
    
    # CONCAVE FUNCTION
    y_concave = -x_pos**2 + 9
    slope_concave = -2 * x_pos  # f'(x) = -2x
    second_deriv_concave = -2  # f''(x) = -2
    
    # Update point
    point_concave.set_data([x_pos], [y_concave])
    
    # Update tangent line
    y_tangent_concave = y_concave + slope_concave * (x_tangent - x_pos)
    tangent_concave.set_data(x_tangent, y_tangent_concave)
    
    # Add slope direction arrow
    if abs(slope_concave) > 0.1:
        arrow_x = [x_pos, x_pos + arrow_scale]
        arrow_y = [y_concave, y_concave + arrow_scale * slope_concave]
        slope_arrow_concave.set_data(arrow_x, arrow_y)
        slope_arrow_concave.set_marker('>')
        slope_arrow_concave.set_markersize(10)
    
    # Update text
    slope_text_concave.set_text(f'Slope = {slope_concave:.1f}')
    slope_text_concave.set_position((x_pos, y_concave + 1.5))
    
    # Update slope history
    slope_history_x_concave.append(x_pos)
    slope_history_y_concave.append(slope_concave)
    slope_line_concave.set_data(slope_history_x_concave, slope_history_y_concave)
    slope_point_concave.set_data([x_pos], [slope_concave])
    
    # Add annotations in slope plots at key moments
    if frame == 0:  # Start
        ax3.annotate('START:\nSlope very\nnegative', xy=(-2.5, -5), fontsize=9, ha='center', color='red')
        ax4.annotate('START:\nSlope very\npositive', xy=(-2.5, 5), fontsize=9, ha='center', color='blue')
    elif frame == n_frames//2:  # Middle
        ax3.annotate('MIDDLE:\nSlope = 0', xy=(0, 0), fontsize=9, ha='center')
        ax4.annotate('MIDDLE:\nSlope = 0', xy=(0, 0), fontsize=9, ha='center')
    elif frame == n_frames-1:  # End
        ax3.annotate('END:\nSlope very\npositive', xy=(2.5, 5), fontsize=9, ha='center', color='blue')
        ax4.annotate('END:\nSlope very\nnegative', xy=(2.5, -5), fontsize=9, ha='center', color='red')
    
    return (point_convex, tangent_convex, slope_arrow_convex, 
            point_concave, tangent_concave, slope_arrow_concave,
            slope_line_convex, slope_point_convex,
            slope_line_concave, slope_point_concave)
anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True, repeat=True)

plt.tight_layout()

# Display the animation
display(HTML(anim.to_jshtml()))
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Generate margin values (yŷ)
margin = np.linspace(-4, 4, 1000)

# Define loss functions
def zero_one_loss(margin):
    return (margin <= 0).astype(int)

def hinge_loss(margin):
    return np.maximum(0, 1 - margin)

def logistic_loss(margin):
    return np.log(1 + np.exp(-margin)) / np.log(2)

def exponential_loss(margin):
    return np.exp(-margin)

def squared_loss(margin):
    return (1 - margin) ** 2

# Calculate 0-1 loss for all comparisons
loss_01 = zero_one_loss(margin)
#%%
plt.figure(figsize=(8, 5))

# Plot losses
plt.plot(margin, loss_01, 'r-', linewidth=3, label=r'0-1 Loss: $\mathbf{1}[y\hat{y} \leq 0]$', alpha=0.8)
plt.plot(margin, hinge_loss(margin), 'g-', linewidth=2, label=r'Hinge Loss: $\mathit{max}\{0, 1-y\hat{y}\}$')

plt.xlabel('Margin (yŷ)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Hinge Loss vs 0-1 Loss', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 4)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Hinge margin')

# Add annotations
plt.annotate('Correct\nClassification', xy=(2, 0.5), fontsize=12, ha='center')
plt.annotate('Incorrect\nClassification', xy=(-2, 0.5), fontsize=12, ha='center')

plt.tight_layout()
plt.show()

print("""
    When margin > 1 (very confident correct prediction):
        1-yŷ becomes negative
        max{0, negative} = 0
        Loss = 0 (no penalty)
    
    When 0 < margin ≤ 1 (correct but not confident enough):
        1-yŷ is positive but small
        max{0, positive} = positive value
        Loss = 1-yŷ (linear penalty)
    
    When margin ≤ 0 (wrong prediction):
        1-yŷ ≥ 1
        max{0, ≥1} = ≥1
        Loss = 1-yŷ (large linear penalty)
""")

print("\033[1mThe hinge loss penalises linearly, but doesn't differentiate between confident examples because it has a\"flat zone\" where the loss equals zero.\033[0m")
plt.figure(figsize=(8, 5))

# Plot losses
plt.plot(margin, loss_01, 'r-', linewidth=3, label=r'0-1 Loss: $\mathbf{1}[y\hat{y} \leq 0]$', alpha=0.8)
plt.plot(margin, logistic_loss(margin), 'g-', linewidth=2, label=r'Logistic Loss: $\frac{1}{log2} log(1 + e^{[-y\hat{y}]})$')

plt.xlabel('Margin (yŷ)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Logistic Loss vs 0-1 Loss', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 4)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.annotate('Smooth approximation', xy=(1, 1.5), fontsize=12, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.tight_layout()
plt.show()


print("""
    When margin > 1 (very confident correct prediction):
        e^(-yŷ) becomes very small (approaching 0)
        log(1 + very small) ≈ log(1) = 0
        Loss ≈ 0 (but never exactly 0, always decreasing)
    
    When 0 < margin ≤ 1 (correct but not confident enough):
        e^(-yŷ) ranges from e^(-1) ≈ 0.37 to e^0 = 1
        log(1 + 0.37) to log(1 + 1) = log(1.37) to log(2)
        Loss = moderate values (smooth decrease as margin increases)
    
    When margin ≤ 0 (wrong prediction):
        e^(-yŷ) ≥ e^0 = 1 (grows for more negative margins)
        log(1 + ≥1) ≥ log(2)
        Loss ≥ 1 (grows without bound for confident wrong predictions)
""")

print("\033[1mThe logistic loss penalises linearly and promotes confident examples.\033[0m")
plt.figure(figsize=(8, 5))

# Plot losses
plt.plot(margin, loss_01, 'r-', linewidth=3, label=r'0-1 Loss: $\mathbf{1}[y\hat{y} \leq 0]$', alpha=0.8)
plt.plot(margin, exponential_loss(margin), 'purple', linewidth=2, label=r'Exponential Loss: $e^{[-yŷ]}$')

plt.xlabel('Margin (yŷ)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Exponential Loss vs 0-1 Loss', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 8)  # Larger range for exponential
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.annotate('Grows exponentially\nfor wrong predictions', 
            xy=(-1.5, 4), fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.7))

plt.tight_layout()
plt.show()


print("""
    When margin > 1 (very confident correct prediction):
        e^(-yŷ) becomes very small (approaching 0)
        Loss = small positive value (decreases exponentially, never reaches 0)
    
    When 0 < margin ≤ 1 (correct but not confident enough):
        e^(-yŷ) ranges from e^(-1) ≈ 0.37 to e^0 = 1
        Loss = moderate values (exponential decrease as margin increases)
    
    When margin ≤ 0 (wrong prediction):
        e^(-yŷ) ≥ e^0 = 1 (grows exponentially for negative margins)
        Loss ≥ 1 (explodes exponentially for confident wrong predictions)
""")

print("\033[1mThe exponential loss penalises super-linearly and promotes confident examples.\033[0m")
plt.figure(figsize=(8, 5))

# Plot losses
plt.plot(margin, loss_01, 'r-', linewidth=3, label=r'0-1 Loss: $\mathbf{1}[y\hat{y} \leq 0]$', alpha=0.8)
plt.plot(margin, squared_loss(margin), 'orange', linewidth=2, label=r'Squared Loss: $(y - \hat{y})^2$')

plt.xlabel('Margin (yŷ)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Squared Loss vs 0-1 Loss', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 6)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='orange', linestyle='--', alpha=0.5)

plt.annotate('Minimum at yŷ=1', xy=(1, 0.2), fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="moccasin", alpha=0.7))

plt.tight_layout()
plt.show()


print("""
When margin > 1 (very confident correct prediction):
    (y-ŷ) becomes negative (since y-ŷ > 1)
    (negative)² = positive value
    Loss = (y-ŷ)² (increases quadratically as confidence grows)

When 0 < margin ≤ 1 (correct but not confident enough):
    (y-ŷ) ranges from (1-0) = 1 to (1-1) = 0
    Loss = (y-ŷ)² (decreases quadratically toward minimum)

When margin ≤ 0 (wrong prediction):
    (y-ŷ) ≥ (1-0) = 1 (grows as margin becomes more negative)
    Loss = (y-ŷ)² (increases quadratically)
""")

print("\033[1mThe squared loss penalises super-linearly and penalises over-confident examples as much as mistakes.\033[0m")

#and finally ...life example
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
    # 'Board Games': ['No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Online Games': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Output': ['Buys', 'Cancels', 'Buys', 'Cancels', 'Cancels', 'Cancels', 'Buys', 'Cancels', 'Cancels', 'Buys']
}
history = []  # Track (w, b, step_name) for visualization

df = pd.DataFrame(data)
print("=" * 60)
print("CUSTOMER TRANSACTION DATA")
print("=" * 60)
print(df.to_string(index=False))
print("\nThis data will be used to train a perceptron to predict customer behavior!")
#%%
# Convert Yes/No to 1/0 and Buys/Cancels to +1/-1
vectorized_data = {
    'Transaction': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
    'Music Download?': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    'Music Streaming?': [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    # 'Board Games': [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    'Online Games': [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    'Output': [1, -1, 1, -1, -1, -1, 1, -1, -1, 1]
}

df_vec = pd.DataFrame(vectorized_data)

print("=" * 80)
print("VECTORIZED DATA")
print("=" * 80)
print("Features:")
print("  • Yes → 1")
print("  • No → 0")
print("\nClass (Output):")
print("  • Buys → +1")
print("  • Cancels → -1")
print()
print(df_vec.to_string(index=False))

# Create feature matrix X and labels y
X = df_vec[['Music Download?', 'Music Streaming?', 'Online Games']].values
y = df_vec['Output'].values

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Labels y shape: {y.shape}")
#%%
# Initialize and First Training Example
# Training data
x_T1 = np.array([0, 1, 1])  # T1
y_T1 = 1

x_T2 = np.array([1, 0, 0])  # T2
y_T2 = -1

# Initialize weights and bias
w = np.array([0.0, 0.0, 0.0])
b = 0.0

print("=" * 80)
print("PERCEPTRON TRAINING: FIRST EXAMPLE")
print("=" * 80)
print(f"Training Examples:")
print(f"  x_T1 = {x_T1}; y_T1 = {y_T1}")
print(f"  x_T2 = {x_T2}; y_T2 = {y_T2}")
print()
print(f"Initialize weights and bias:")
print(f"  w⁽⁰⁾ = {w}; b⁽⁰⁾ = {b}")
print()

# Iteration 1 - Process T1
print(f"ITERATION 1 - Processing x_T1:")
a = np.dot(w, x_T1) + b
print(f"  a = w·x + b = {np.dot(w, x_T1)} + {b} = {a}")

condition = y_T1 * a <= 0
print(f"  y_T1 · a = {y_T1} × {a} = {y_T1 * a} ≤ 0? {condition} → {'UPDATE' if condition else 'NO UPDATE'}")

if condition:
    w = w + y_T1 * x_T1
    b = b + y_T1
    print(f"  w⁽¹⁾ = w⁽⁰⁾ + y_T1 × x_T1 = {w - y_T1 * x_T1} + {y_T1} × {x_T1} = {w}")
    print(f"  b⁽¹⁾ = b⁽⁰⁾ + y_T1 = {b - y_T1} + {y_T1} = {b}")

print(f"\nFinal: w⁽¹⁾ = {w}, b⁽¹⁾ = {b}")
history.append((w.copy(), b, "After Example 1"))
#%%
# Second Training Example
print("=" * 80)
print("PERCEPTRON TRAINING: SECOND EXAMPLE")
print("=" * 80)
print(f"Current state:")
print(f"  w⁽¹⁾ = {w}; b⁽¹⁾ = {b}")
print()

# Process T2
print(f"Processing x_T2:")
a = np.dot(w, x_T2) + b
print(f"  a = Σw_d×x_d + b = {w[0]}×{x_T2[0]} + {w[1]}×{x_T2[1]} + {w[2]}×{x_T2[2]} + {b}")
print(f"    = {w[0]*x_T2[0]} + {w[1]*x_T2[1]} + {w[2]*x_T2[2]} + {b} = {a}")

condition = y_T2 * a <= 0
print(f"  y_T2 · a = ({y_T2}) × {a} = {y_T2 * a} ≤ 0? {condition} → {'UPDATE' if condition else 'NO UPDATE'}")

if condition:
    w_old = w.copy()
    b_old = b
    w = w + y_T2 * x_T2
    b = b + y_T2
    print(f"  w⁽²⁾ = w⁽¹⁾ + y_T2 × x_T2 = {w_old} + ({y_T2}) × {x_T2} = {w}")
    print(f"  b⁽²⁾ = b⁽¹⁾ + y_T2 = {b_old} + ({y_T2}) = {b}")

print(f"\nFinal: w⁽²⁾ = {w}, b⁽²⁾ = {b}")
history.append((w.copy(), b, "After Example 2"))
#%%
# Iteration 2 for Both Examples
print("=" * 80)
print("ITERATION 2 for T1 and T2 Examples")
print("=" * 80)
print(f"Current weights: w⁽²⁾ = {w}; b⁽²⁾ = {b}")
print()

# T1
print("x_T1:")
a_T1 = np.dot(w, x_T1) + b
print(f"  a = Σw_d×x_d + b = ({w[0]})×{x_T1[0]} + {w[1]}×{x_T1[1]} + {w[2]}×{x_T1[2]} + {b}")
print(f"    = {w[0]*x_T1[0]} + {w[1]*x_T1[1]} + {w[2]*x_T1[2]} + {b} = {a_T1}")

condition_T1 = y_T1 * a_T1 <= 0
print(f"  y_T1 · a = {y_T1} × {a_T1} = {y_T1 * a_T1} ≤ 0? {condition_T1} → {'UPDATE' if condition_T1 else 'NO UPDATE'}")

# T2  
print("\nx_T2:")
a_T2 = np.dot(w, x_T2) + b
print(f"  a = Σw_d×x_d + b = ({w[0]})×{x_T2[0]} + {w[1]}×{x_T2[1]} + {w[2]}×{x_T2[2]} + {b}")
print(f"    = {w[0]*x_T2[0]} + {w[1]*x_T2[1]} + {w[2]*x_T2[2]} + {b} = {a_T2}")

condition_T2 = y_T2 * a_T2 <= 0
print(f"  y_T2 · a = ({y_T2}) × {a_T2} = {y_T2 * a_T2} ≤ 0? {condition_T2} → {'UPDATE' if condition_T2 else 'NO UPDATE'}")

if not condition_T1 and not condition_T2:
    print(f"\nCONVERGENCE..Both examples classified correctly!")
    print(f"Final weights: w = {w}; b = {b}")
else:
    print(f"\nStill learning... need more iterations")
history.append((w.copy(), b, "After Iteration 2"))
#%%
#matplotlib inline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', message='Glyph.*missing from font.*')

# Training data points
x_T1 = np.array([0, 1, 1])
y_T1 = 1
x_T2 = np.array([1, 0, 0])
y_T2 = -1

# Add initial state to history if not already there
if len(history) == 0 or not np.array_equal(history[0][0], [0, 0, 0]):
    history.insert(0, (np.array([0.0, 0.0, 0.0]), 0.0, "Initial State"))

def plot_simple_hyperplane(ax, w, b, title):
    """Hyperplane plotting"""
    
    # Plot data points first
    ax.scatter(*x_T1, color='green', s=200, label=f'T1 (+1)', alpha=0.8)
    ax.scatter(*x_T2, color='red', s=200, label=f'T2 (-1)', alpha=0.8)
    
    # Only plot hyperplane if weights are meaningful
    if not np.allclose(w, 0):
        
        x_range = np.linspace(-1, 2, 10)
        y_range = np.linspace(-1, 2, 10)
        
        if abs(w[2]) > 1e-10:  # Can solve for z
            XX, YY = np.meshgrid(x_range, y_range)
            ZZ = -(w[0]*XX + w[1]*YY + b) / w[2]
            
            # Show hyperplane
            ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='lightcyan', edgecolor='lightsteelblue', linewidth=0.3)
        
        elif abs(w[1]) > 1e-10:  # Can solve for y (when w[2] = 0)
            XX, ZZ = np.meshgrid(x_range, np.linspace(-1, 2, 10))
            YY = -(w[0]*XX + w[2]*ZZ + b) / w[1]
            ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='lightcyan', edgecolor='lightsteelblue', linewidth=0.3)
        
        elif abs(w[0]) > 1e-10:  # Can solve for x (when w[1] = w[2] = 0)
            YY, ZZ = np.meshgrid(y_range, np.linspace(-1, 2, 10))
            XX = -(w[1]*YY + w[2]*ZZ + b) / w[0]
            ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='lightcyan', edgecolor='lightsteelblue', linewidth=0.3)
    else:
        print(f"No hyperplane for {title} (all weights zero)")
        ax.text(0.5, 0.5, 0.5, 'No hyperplane\n(w = 0)', 
               fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # Labels and formatting
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title(f'{title}\nw = [{w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}], b = {b:.1f}', fontsize=10)
    ax.legend()
    
    # Set limits
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, 1.5])
#%%
# Create figure
fig = plt.figure(figsize=(16, 8))

# Create subplots
n_steps = len(history)
for i, (w, b, step_name) in enumerate(history):
    ax = fig.add_subplot(1, n_steps, i+1, projection='3d')
    plot_simple_hyperplane(ax, w, b, step_name)
    ax.view_init(elev=5, azim=50)

plt.tight_layout()
plt.show()

