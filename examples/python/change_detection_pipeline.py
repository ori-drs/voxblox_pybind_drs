# Voxel grid filter pipeline
from easydict import EasyDict
import numpy as np
import open3d as o3d
import os 
import yaml
import time 

def load_config(config_file: str):
    return EasyDict(yaml.safe_load(open(config_file)))

def write_config(config: EasyDict, filename: str):
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

class ChangeDetectionPipeline:
    """Abstract class that defines a Pipeline, derived classes must implement the dataset and config
    properties."""
    def __init__(self, mesh1, mesh2, config_file: str):
        self._mesh1 = mesh1
        self._mesh2 = mesh2
        self._config = load_config(config_file)
        self._voxel_size = self._config.voxel_size

    def run(self):
        self._run_voxel_grid_pipeline()
        self._write_ply()
        self._write_cfg()

    def _run_voxel_grid_pipeline(self):
        times = []
        tic = time.perf_counter_ns()
        # Apply voxel grid filter on both map 1 and map 2
        self._voxel_grid1 = self.voxel_grid_from_triangle_mesh(self._mesh1, self._voxel_size)
        self._voxel_grid2 = self.voxel_grid_from_triangle_mesh(self._mesh1, self._voxel_size) 
        # Get static, dynamic and change grids
        diff_grid, dynamic_grid, static_grid = \
                        self.voxel_grid_change_detection(self._voxel_grid1, self._voxel_grid2)
        # Get meshes where output grids are occupied
        toc = time.perf_counter_ns()
        times.append(toc - tic)
        # TODO (joseph@robots.ox.ac.uk)
        # right now, dynamic and static meshes are grids on mesh1, 
        #   should make it grids on both summed.
       
        self._res = {
                    "mesh1": self._mesh1,
                    "mesh2": self._mesh2,
                    "grid1": self._voxel_grid1,
                    "grid2": self._voxel_grid2,
                    "dynamic_mesh": self.get_occupied_mesh(self._mesh1, dynamic_grid),
                    "static_mesh": self.get_occupied_mesh(self._mesh1, static_grid) 
                }



    def voxel_grid_filter(self, mesh, voxel_size):
        # Create a grid with the specified voxel size
        voxel_grid = np.zeros((mesh.shape[0]//voxel_size, mesh.shape[1]//voxel_size, mesh.shape[2]//voxel_size))
        # Iterate through each voxel in the grid
        print("voxelization...")
        for i in range(voxel_grid.shape[0]):
            for j in range(voxel_grid.shape[1]):
                for k in range(voxel_grid.shape[2]):
                    # Calculate the average value of the TSDF values within the voxel
                    voxel_mean = np.mean(mesh[i*voxel_size:(i+1) * voxel_size, j 
                        * voxel_size:(j+1) * voxel_size, k * voxel_size:(k+1) * 
                        voxel_size])
                    # Set the value of the voxel in the grid to the calculated mean
                    voxel_grid[i,j,k] = voxel_mean
    
        return voxel_grid

    def voxel_grid_from_triangle_mesh(self, mesh, voxel_size):
        # fit to unit cube
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                center=mesh.get_center())
        print('voxelization...')
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, \
                                                                    voxel_size)
        return voxel_grid

    def draw_geometries(self):
        # print("Try to render a mesh with normals (exist: " + \
        # str(mesh.has_vertex_normals()) + ") and colors (exist: " + \
        # str(mesh.has_vertex_colors()) + ")")

        if (self._res["dynamic_mesh"].has_vertex_colours()== False):
            print("Computing normal and rendering it.")
            self._res["dynamic_mesh"].compute_vertex_normals()
            self._res["static_mesh"].compute_vertex_normals()
        print("Dynamic mesh.")
        o3d.visualization.draw_geometries(self._res["dynamic_mesh"])
        print("Static mesh")
        o3d.visualization.draw_geometries(self._res["static_mesh"])

    

    def _write_ply(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        static_filename = os.path.join(self._config.out_dir, self._map_name) + "_static.ply"
        dynamic_filename = os.path.join(self._config.out_dir, self._map_name) + "_dynamic.ply"
        o3d.io.write_triangle_mesh(static_filename, self._res["static_mesh"])
        o3d.io.write_triangle_mesh(dynamic_filename, self._res["dynamic_mesh"])

    def _write_cfg(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".yml"
        write_config(dict(self._config), filename)

    def voxel_grid_change_detection(self, grid1, grid2):
        # Create a new grid to store the difference between the two grids
        diff_grid = np.zeros(grid1.shape)
        dynamic_grid = np.zeros(grid1.shape)
        static_grid = np.zeros(grid1.shape)
        # Iterate through each voxel in both grids
        for i in range(grid1.shape[0]):
            for j in range(grid1.shape[1]):
                for k in range(grid1.shape[2]):
                    # Calculate the absolute difference between the two voxels
                    voxel_diff = abs(grid1[i,j,k] - grid2[i,j,k])
                    # Set the value of the voxel in the difference grid to the calculated difference
                    diff_grid[i,j,k] = voxel_diff
                    # If the difference is greater than the changed threshold, set the voxel in the changed grid to 1
                    if voxel_diff > self._changed_threshold:
                        dynamic_grid[i,j,k] = 1
                        # Otherwise, set the voxel in the static grid to 1
                    else:
                        static_grid[i,j,k] = 1

        # Return the difference grid, equivalent to dynamic map
        return diff_grid, dynamic_grid, static_grid

    def get_occupied_mesh(self, mesh, grid):
        # Create a new mesh with the same dimensions as the original
        occupied_mesh = np.zeros(mesh.shape)
        # Iterate through each voxel in the grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    # If the voxel is occupied (not empty), copy the corresponding values from the original mesh
                    if grid[i,j,k] != 0:
                        occupied_mesh[i*self._voxel_size:(i+1)*self._voxel_size, 
                            j*self._voxel_size:(j+1)*self._voxel_size, k*self._voxel_size:(k+1)*self._voxel_size] = mesh[i*voxel_size:(i+1)*self._voxel_size, j*self._voxel_size:(j+1)*self._voxel_size, 
                            k*self._voxel_size:(k+1)*self._voxel_size]

        # Return the occupied mesh
        return occupied_mesh

