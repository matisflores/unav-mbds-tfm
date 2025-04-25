import matplotlib.pyplot as plt
import numpy as np

from mot.Trackers import ParticleTracker
from motpy.core import Detection
from utils.Config import Config

def point_to_box(center, w, h):
    return np.array([center[0] - w, center[1] - h, center[0] + w,  center[1] + h])

config = Config()
config.load('config.ini')

num_steps = 7
dt = 1

# Parameters for the first dot
initial_position1 = np.array([0, 0])
velocity1 = np.array([1, 1])

tracker = ParticleTracker({'dt':dt}, initial_position1, point_to_box(initial_position1, 20, 20))

# Parameters for the second dot
initial_position2 = tracker._center
#velocity2 = np.array([-1, 1])




# Initialize arrays to store position history
positions1 = np.zeros((num_steps, 2))
positions1[0] = initial_position1

positions2 = np.zeros((num_steps, 2))
positions2[0] = initial_position2

positions3 = np.zeros((num_steps, 2))
positions3[0] = initial_position2

# Simulate the movement
for i in range(1, num_steps):
    positions1[i] = positions1[i-1] + velocity1 * dt
    print('Original', positions1[i])
    tracker.predict()
    positions2[i] = tracker._center
    print('Prediction', positions2[i])
    tracker.update(Detection(point_to_box(positions1[i], 20, 20), 1., 0))
    positions3[i] = tracker._center
    print('Updated', positions3[i])
    print('Error', tracker.error())

    '''
    # Plot the path
    plt.figure(figsize=(8, 8))
    plt.plot(positions1[:i+1, 0], positions1[:i+1, 1], marker='o', linestyle='-', color='b', label='Dot 1')
    plt.plot(positions2[:i+1, 0], positions2[:i+1, 1], marker='o', linestyle='-', color='r', label='Dot 2')
    plt.title(f'Dots Moving with Constant Velocity (step={i})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Save the current step plot
    #plt.savefig(f'plot_step_{i}.png')
    plt.show()
    #plt.close()
    '''

plt.figure(figsize=(8, 8))
plt.plot(positions1[:, 0], positions1[:, 1], marker='o', linestyle='dotted', color='b', label='Dot 1')
plt.plot(positions2[:, 0], positions2[:, 1], marker='o', linestyle='dotted', color='r', label='Dot 2')
plt.plot(positions3[:, 0], positions3[:, 1], marker='o', linestyle='dotted', color='g', label='Dot 3')
plt.title('Dots Moving with Constant Velocity')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()