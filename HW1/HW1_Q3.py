import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = np.linspace(1, 10, 2000)
c = 0.25
robotPos = np.array([np.cos(t), np.sin(t), np.sin(t)])
cameraPos = np.array([np.cos(t) + np.cos(10*t)*c, np.sin(t) + np.sin(10*t)*c, np.sin(t)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(robotPos[0], robotPos[1], robotPos[2], label='Robot Position', color='blue')
ax.plot3D(cameraPos[0], cameraPos[1], cameraPos[2], label='Camera Position', color='black')
ax.scatter3D(0, 0, 0, label='Origin', color='red', marker='o', s = 50)
ax.legend()
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Robot and Camera Position Plot')
plt.show()