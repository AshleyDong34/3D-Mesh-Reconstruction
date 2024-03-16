import os
import polyscope as ps
import numpy as np
from skimage import measure
from ReconstructionFunctions import load_off_file, compute_RBF_weights, evaluate_RBF, polyharmonic, Wendland, biharmonic


if __name__ == '__main__':
    ps.init()

    inputPointNormals, _ = load_off_file(os.path.join(os.getcwd(), 'data', 'dragon-3000.off'))
    inputPoints = inputPointNormals[:, 0:3]
    inputNormals = inputPointNormals[:, 3:6]

    # normalizing point cloud to be centered on [0,0,0] and between [-0.9, 0.9]
    inputPoints -= np.mean(inputPoints, axis=0)
    min_coords = np.min(inputPoints, axis=0)
    max_coords = np.max(inputPoints, axis=0)
    scale_factor = 0.9 / np.max(np.abs(inputPoints))
    inputPoints = inputPoints * scale_factor

    ps_cloud = ps.register_point_cloud("Input points", inputPoints)
    ps_cloud.add_vector_quantity("Input Normals", inputNormals)


    # Parameters
    gridExtent = 1 #the dimensions of the evaluation grid for marching cubes
    res = 50 #the resolution of the grid (number of nodes)

    # Generating and registering the grid
    gridDims = (res, res, res)
    bound_low = (-gridExtent, -gridExtent, -gridExtent)
    bound_high = (gridExtent, gridExtent, gridExtent)
    ps_grid = ps.register_volume_grid("Sampled Grid", gridDims, bound_low, bound_high)

    X, Y, Z = np.meshgrid(np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res),
                          np.linspace(-gridExtent, gridExtent, res), indexing='ij')

    #the list of points to be fed into evaluate_RBF
    xyz = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    ##########################
    ## you code of computation and evaluation goes here
    ##
    ##
    N = len(inputPoints)  # Number of original input points
    initial_subset_size = 200  # Starting number of points
    centre_indices = np.random.choice(N, initial_subset_size, replace=False)

    epsilon = 0.01  # This is an example value, you might need to adjust it
    RBFFunction = lambda r: Wendland(r)

    # Compute RBF weights using the provided inputPoints, inputNormals, and RBFFunction
    w, RBFCentres, a_value = compute_RBF_weights(inputPoints, inputNormals, RBFFunction,epsilon,RBFCentreIndices=[], useOffPoints=True, sparsify = True,l = -1)

    # Evaluate the RBF at the grid points
    RBFValues = evaluate_RBF(xyz, RBFCentres, RBFFunction, w, l = -1, a=a_value)
    
    print("RBFValues: ", RBFValues)
    print(np.count_nonzero(RBFValues==0))
    print(len(RBFValues))
    print("weights: ", w)
    ##
    ##
    #########################

    #fitting to grid shape again
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
