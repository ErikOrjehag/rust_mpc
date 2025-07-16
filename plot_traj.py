import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("trajectory.csv")
plt.plot(df['rx'], df['ry'], marker='o', color='red', label='Reference Trajectory')
plt.plot(df['x'], df['y'], marker='o', color='blue', label='Robot Trajectory')
plt.title("Robot Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.show()