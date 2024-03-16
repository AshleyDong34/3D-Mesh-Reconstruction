README for RBF Interpolation and Visualization Script
Overview

This script performs Radial Basis Function (RBF) interpolation on a 3D point cloud, specifically using a dataset of a bunny with 500 points. The script iteratively selects a subset of points, computes RBF weights, evaluates the RBF on a grid, and visualizes the resulting mesh using marching cubes and Polyscope.

Requirements
Python 3.x
Numpy
Scikit-image
Polyscope
Additional dependencies as required by ReconstructionFunctions.py.
Functionality
Point Cloud Loading and Normalization: The script begins by loading a 3D point cloud from an OFF file (in this case, bunny-500.off). The point cloud is normalized to be centered around [0, 0, 0] and scaled within [-0.9, 0.9].

Initialization of Polyscope: Polyscope is used for visualization. It's initialized and the point cloud is registered for display.

Grid Setup for Marching Cubes: A grid is set up for the marching cubes algorithm, which will be used later to visualize the implicit surface formed by the RBF.

RBF Interpolation:

A subset of points is randomly chosen as centers for the RBF interpolation.
Radial basis function weights are computed using these centers.
The RBF is evaluated over all points in the point cloud, including off-surface points generated using normal information.
Residuals are calculated to determine the fitting error at each iteration.
Visualization:

Once the residuals fall below a specified threshold, the RBF is evaluated on the grid points.
Marching cubes algorithm is applied to extract a mesh from the scalar field defined by the RBF.
This mesh is visualized using Polyscope.
Iterative Process: The script iteratively augments the set of RBF centers with points where the fitting error is largest, recalculates the weights and residuals, and updates the visualization until a specified tolerance level is reached.

Running the Script
To run the script, ensure all dependencies are installed and execute the script in a Python environment. Ensure the bunny-500.off file and ReconstructionFunctions.py are in the correct path as referenced in the script.

Customization
You can customize various parameters such as:

The RBF function (e.g., polyharmonic, Wendland, biharmonic).
The tolerance for error improvement.
The initial subset size for RBF centers.
The file path for different point cloud datasets.