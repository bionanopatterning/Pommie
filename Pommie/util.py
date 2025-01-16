import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data: theta and phi
np.random.seed(0)
theta = np.random.vonmises(mu=0, kappa=4, size=1000)
phi = np.random.uniform(0, 2*np.pi, 1000)


def plot_angle_histogram_on_unit_sphere(theta, phi):
    hist, theta_edges, phi_edges = np.histogram2d(theta, phi, bins=50)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers)

    # Convert to Cartesian coordinates for sphere surface
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    # Flatten the arrays
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    hist_flat = hist.ravel()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each bar
    bar_length_factor = 0.1  # Factor to scale the length of the bars
    for xi, yi, zi, hi in zip(x_flat, y_flat, z_flat, hist_flat):
        # Calculate the end point of the bar
        length = bar_length_factor * hi
        xe, ye, ze = xi + length * xi, yi + length * yi, zi + length * zi

        # Plot a line from the sphere's surface to the end point
        ax.plot([xi, xe], [yi, ye], [zi, ze], color='b')

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("Histogram of phi, theta coordinates of transforms in dataset.")
    plt.show()
