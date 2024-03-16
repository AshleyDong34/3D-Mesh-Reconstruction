import os
import numpy as np
from scipy.spatial import distance
from scipy.linalg import lu_factor, lu_solve, lstsq
from functools import partial
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.sparse import dok_matrix


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=[], useOffPoints=True, sparsify=False, l=-1):
    N = len(inputPoints)
    
    # Generate off-surface points
    off_surface_points_plus = inputPoints + epsilon * inputNormals
    off_surface_points_minus = inputPoints - epsilon * inputNormals
    all_data_points = np.vstack((inputPoints, off_surface_points_plus, off_surface_points_minus))
    
    # Construct vector b
    b = np.concatenate([np.zeros(N), np.full(N, epsilon), np.full(N, -epsilon)])
    
    # Select RBF centres
    if len(RBFCentreIndices) > 0:
        subset_centres = inputPoints[RBFCentreIndices]
        if useOffPoints:
            subset_centres_plus = subset_centres + epsilon * inputNormals[RBFCentreIndices]
            subset_centres_minus = subset_centres - epsilon * inputNormals[RBFCentreIndices]
            RBFCentres = np.vstack((subset_centres, subset_centres_plus, subset_centres_minus))
        else:
            RBFCentres = subset_centres
    else:
        RBFCentres = all_data_points if useOffPoints else inputPoints

    if sparsify:
        # Sparse matrix construction using KDTree and sparse distance matrix
        tree_data = cKDTree(all_data_points)
        tree_centres = cKDTree(RBFCentres)
        max_distance = 2 * epsilon
        # Create a sparse distance matrix
        sparse_dist_matrix = tree_data.sparse_distance_matrix(tree_centres, max_distance, output_type='coo_matrix').tocsr()

        # Evaluate the RBF values using sparse matrix operations
        rbf_values = RBFFunction(sparse_dist_matrix.data)
        sparse_dist_matrix.data = rbf_values

        # A is the sparse distance matrix
        A = sparse_dist_matrix.tocsr()
 
    else:
        # Dense matrix construction
        pairwise_distances = distance.cdist(all_data_points, RBFCentres)
        A = RBFFunction(pairwise_distances)

    if sparsify:
        w = spsolve(A, b)
        a = []  # Polynomial terms are not considered in sparse mode 
    elif l > -1:
        # Generate polynomial terms
        indices = generate_index_triplets(l)
        Q = construct_Q_matrix(all_data_points, indices)
        A = np.block([[A, Q], [Q.T, np.zeros((len(indices), len(indices)))]])
        b = np.concatenate((b, np.zeros(len(indices))))
            
        LU, piv = lu_factor(A)
        w_a = lu_solve((LU, piv), b)
        w, a = np.split(w_a, [len(RBFCentres)])
    else:
        # Solve the system
        if A.shape[0] == A.shape[1]:
            # LU decomposition for square matrix
            LU, piv = lu_factor(A)
            w = lu_solve((LU, piv), b)
        else:
            # Least squares for non-square matrix
            w, _, _, _ = lstsq(A, b)
        a = []

    return w, RBFCentres, a



def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):
    tree = cKDTree(centres)
    values = np.zeros(len(xyz))
    
    if len(centres) > 4500*3:
        support_radius = 0.1
    else:
        # essentially no sparsity, used for smaller point clouds
        support_radius = 1000
        
    # Evaluate the RBF only for points within the support radius
    for i, point in enumerate(xyz):
        indices = tree.query_ball_point(point, support_radius)
        if indices:
            distances = np.linalg.norm(centres[indices] - point, axis=1)
            rbf_values = RBFFunction(distances)
            values[i] = np.dot(rbf_values, w[indices])

    # Add polynomial contribution if applicable
    if l > -1:
        indices = generate_index_triplets(l)
        poly_vals = polynomial_contribution(xyz, a, indices)
        values += poly_vals

    return values


def generate_index_triplets(l):
    i, j, k = np.meshgrid(np.arange(l+1), np.arange(l+1), np.arange(l+1), indexing='ij')
    mask = (i + j + k) <= l
    return np.vstack([i[mask], j[mask], k[mask]]).T

def construct_Q_matrix(all_points, indices):
    Q = []
    for point in all_points:
        row = [point[0]**i * point[1]**j * point[2]**k for (i, j, k) in indices]
        Q.append(row)
    return np.array(Q)

def polynomial_contribution(xyz, a, indices):
    poly_values = np.zeros(len(xyz))
    for i, (x, y, z) in enumerate(xyz):
        poly_values[i] = sum(a[k] * x**index[0] * y**index[1] * z**index[2] for k, index in enumerate(indices))
    return poly_values

def biharmonic(r):
    return r

def polyharmonic(r):
    return r**3

def wendland_function(r, beta=0.5):
    positive_part = np.maximum(1 - beta * r, 0)
    return (1/12) * (positive_part**3) * (1 - 3 * beta * r)

fixed_beta = 0.5
Wendland = partial(wendland_function, beta=fixed_beta)


