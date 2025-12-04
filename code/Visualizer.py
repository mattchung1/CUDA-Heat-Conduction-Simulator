import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Settings (Must match your C++ settings!)
gridSize = 258  # (numPoints + 2)
dtype = np.float64 # Matches C++ 'double'

# Load the entire file at once
try:
    raw_data = np.fromfile("sim_data.bin", dtype=dtype)
    # Reshape: -1 means "calculate how many frames based on file size"
    frames = raw_data.reshape(-1, gridSize, gridSize)
except FileNotFoundError:
    print("Run the simulation first!")
    exit()

print(f"Loaded {len(frames)} frames.")

# Setup Plot
fig, ax = plt.subplots()
heatmap = ax.imshow(frames[0], cmap='hot', vmin=0, vmax=100)
plt.colorbar(heatmap)
title = ax.set_title("Iteration 0")

# Animation Function
def update(frame_idx):
    heatmap.set_data(frames[frame_idx])
    title.set_text(f"Frame {frame_idx}")
    return heatmap, title

# Play
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
plt.show()