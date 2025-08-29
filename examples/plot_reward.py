import matplotlib.pyplot as plt
import numpy as np

ttc_x = np.linspace(-100, 100, 1000)
ttc_y = np.linspace(-100, 100, 1000)
r_x = np.where(
    ttc_x <= 0,
    1.0 / (1.0 + np.exp(-4 - 0.1 * ttc_x)),
    -1.0 / (1.0 + np.exp(4 - 0.1 * ttc_x)),
)
r_y = np.where(
    ttc_y <= 0,
    1.0 / (1.0 + np.exp(-4 - 0.1 * ttc_y)),
    -1.0 / (1.0 + np.exp(4 - 0.1 * ttc_y)),
)
plt.plot(ttc_x, r_x, label="Reward function for TTC_x")
plt.plot(ttc_y, r_y, label="Reward function for TTC_y")
plt.xlabel("Time to Collision")
plt.ylabel("Reward")

plt.legend()
plt.show()


# Create Meshgrid of TTC
dx = np.linspace(-100, 100, 1000)
dy = np.linspace(-100, 100, 1000)
kx = 0.001
ky = 0.001
r = np.exp(-kx * dx**2 - ky * dy**2)

X, Y = np.meshgrid(dx, dy)
Z = np.exp(-kx * X**2 - ky * Y**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")  # Set colormap to viridis

ax.set_xlabel("dx")
ax.set_ylabel("dy")
ax.set_zlabel("Reward")

plt.show()
