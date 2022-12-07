#vdbf_pipeline
from functools import reduce
import os
import time

from easydict import EasyDict
import open3d as o3d
from tqdm import trange
import yaml
import vdbfusion
import numpy as np

def load_config(config_file: str):
    return EasyDict(yaml.safe_load(open(config_file)))


def write_config(config: EasyDict, filename: str):
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


class VDBPipeline:
    """Abstract class that defines a Pipeline, derived classes must implement the dataset and config
    properties."""

    def __init__(self, dataset, config_file: str, jump: int, n_scans: int, map_name: str):
        self._dataset = dataset
        self.vis = True
        self._config = load_config(config_file)
        self._n_scans = len(dataset) if n_scans == -1 else n_scans
        self._jump = jump
        self._map_name = map_name
        self._res = {}
        self._voxel_size = self._config.voxel_size
        self._sdf_trunc = self._config.sdf_trunc
        self._space_carving = self._config.space_carving
        self._sigma = self._config.sigma


    def run(self):
        self._run_vdb_pipeline()
        self._write_ply()
        self._write_cfg()
        self._print_tim()

    def draw_mesh(self):
        o3d.visualization.draw_geometries([self._res["mesh"]])

    def update_mesh(self):
        o3d.visualization.update_geometries([self._res["mesh"]])

    def __len__(self):
        return len(self._dataset)

    def _run_vdb_pipeline(self):
        #vdbf implementation
        vdb_volume = vdbfusion.VDBVolume(self._voxel_size, self._sdf_trunc, self._space_carving)
        for scan, pose in self._dataset:
            vdb_volume.integrate(scan, pose) # PROBLEM HERE, skip first pose !
                                # lambda sdf:1.0 
                                # if sdf < self._voxel_size else np.exp(-self._sigma * (sdf- self._voxel_size) ** 2.0),)
        #extract triangle mesh np arrays from internal map
        vertice, triangles = vdb_volume.extract_triangle_mesh()

    def _write_ply(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".ply"
        o3d.io.write_triangle_mesh(filename, self._res["mesh"])

    def _write_cfg(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".yml"
        write_config(dict(self._config), filename)

    def _print_tim(self):
        total_time_ns = reduce(lambda a, b: a + b, self._res["times"])
        total_time = total_time_ns * 1e-9
        total_scans = self._n_scans
        print(f"Avg FPS  = {total_scans / total_time} [Hz]")
        print(f"Total time spent integrating= {total_time} [seconds]")

    @staticmethod
    def _get_o3d_mesh(tsdf_volume):
        print("Meshing...")
        vertices, triangles = tsdf_volume.extract_triangle_mesh()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        return mesh