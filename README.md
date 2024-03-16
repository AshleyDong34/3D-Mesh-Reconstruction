# 3D-Mesh-Reconstruction

We are using radial basis functions to recreate a 3D mesh from point clouds, this may be from scans etc.
There are several different things that are explored here too, eg. adding a global polynomial for shapes that has certain characterisics.
Centre reduction and a script to iteratively add centres to create a minimum error mesh.

There were attempts to create sparse matrixes to be able to cater to larger point clouds. however that would be too time consuming but kd trees were attempted to be used with 
little success, it did save some computer memory and was able to create some pixelated meshes when the point neighbours were reduced for the radal basis function.
