import os
import numpy as np
import polyscope as ps
from skimage import measure
from ReconstructionFunctions import load_off_file, compute_RBF_weights, evaluate_RBF, polyharmonic, Wendland, biharmonic

def calculate_residuals(inputPoints, RBFValues, epsilon, N):
    # Reshape RBFValues to 3 sections
    RBF_original, RBF_plus, RBF_minus = np.split(RBFValues, [N, 2*N])

    # Calculate residuals for each section
    residuals = np.zeros(N * 3)
    residuals[:N] = -RBF_original  # Residuals for original points
    residuals[N:2*N] = epsilon - RBF_plus
    residuals[2*N:] = -epsilon - RBF_minus

    return residuals

def main():
    ps.init()
    inputPointNormals, _ = load_off_file(os.path.join(os.getcwd(), 'data', 'bunny-500.off'))
    inputPoints = inputPointNormals[:, 0:3]
    inputNormals = inputPointNormals[:, 3:6]
    N = len(inputPoints)  # Number of original input points

    # Normalizing point cloud
    inputPoints -= np.mean(inputPoints, axis=0)
    scale_factor = 0.9 / np.max(np.abs(inputPoints))
    inputPoints = inputPoints * scale_factor
   
    ps_cloud = ps.register_point_cloud("Input points", inputPoints)
    ps_cloud.add_vector_quantity("Input Normals", inputNormals)

    gridExtent = 1 #the dimensions of the evaluation grid for marching cubes
    res = 50
    
    gridDims = (res, res, res)
    bound_low = (-gridExtent, -gridExtent, -gridExtent)
    bound_high = (gridExtent, gridExtent, gridExtent)
    ps_grid = ps.register_volume_grid("Sampled Grid", gridDims, bound_low, bound_high)

    X, Y, Z = np.meshgrid(np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res), indexing='ij')

    #the list of points to be fed into evaluate_RBF
    xyz = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))


    epsilon = 0.001
    RBFFunction = lambda r: polyharmonic(r)
    tolerance = 0.0001  # Tolerance for error improvement
    max_iterations = 200  # Maximum number of iterations
    initial_subset_size = 250  # Starting number of points

    # Starting with a random subset of points
    np.random.seed(0)
    centre_indices = np.random.choice(N, initial_subset_size, replace=False)
    used_indices_set = set(centre_indices)

    off_surface_points_plus = inputPoints + epsilon * inputNormals
    off_surface_points_minus = inputPoints - epsilon * inputNormals

    all_data_points = np.vstack((inputPoints, off_surface_points_plus, off_surface_points_minus))

    for iteration in range(max_iterations):
        w, RBFCentres, _ = compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=centre_indices, useOffPoints=True, l=-1)

        # Evaluate the RBF
        RBFValues = evaluate_RBF(all_data_points, RBFCentres, RBFFunction, w, l=-1)

        # Calculate residuals
        residuals = calculate_residuals(inputPoints, RBFValues, epsilon, N)

        # Find the point with the maximum residual not already in the center indices
        max_residual = 0
        max_residual_point = -1
        for i in range(len(residuals)):
            point_idx = i // 3  # Get the original point index from the residual index
            if point_idx not in used_indices_set and abs(residuals[i]) > max_residual:
                max_residual = abs(residuals[i])
                max_residual_point = point_idx

        print(f"Iteration {iteration}: Max Residual = {max_residual}")


        if max_residual < 0.0001:
            RBFValues = evaluate_RBF(xyz, RBFCentres, RBFFunction, w, l = -1)
            RBFValues = np.reshape(RBFValues, X.shape)
            
            # Registering the grid representing the implicit function
            ps_grid.add_scalar_quantity("Implicit Function", RBFValues, defined_on='nodes',
                                        datatype="standard", enabled=True)

            # Computing marching cubes and realigning result to sit on point cloud exactly
            vertices, faces, _, _ = measure.marching_cubes(RBFValues, spacing=(
                2.0 * gridExtent / float(res - 1), 2.0 * gridExtent / float(res - 1), 2.0 * gridExtent / float(res - 1)),
                                                        level=0.0)
            vertices -= gridExtent
            ps.register_surface_mesh("Marching-Cubes Surface", vertices, faces)

            ps.show()
            
            
        # Check for stopping condition
        if max_residual < tolerance:
            break

        # Add the point with the maximum residual to the set
        if max_residual_point != -1:
            centre_indices = np.append(centre_indices, max_residual_point)
            used_indices_set.add(max_residual_point)

if __name__ == "__main__":
    main()