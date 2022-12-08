#!/usr/bin/env python3
# @file      mickie_pipeline.py
#
# Copyright (c) 2021 Ignacio Vizzo, all rights reserved
import argh

from mickie import MickieDataset as Dataset
from tsdf_pipeline import TSDFPipeline
from vdbf_pipeline import VDBPipeline
from change_detection_pipeline import ChangeDetectionPipeline


def main(
    data_source1: str, 
    data_source2: str,
    config: str = "/home/joe/git/voxblox_pybind_drs/examples/python/config/mickie.yaml",
    #config: str = "config/mickie.yaml",
    n_scans: int = -1,
    jump: int = 1,
    visualize: bool = True,
    vdb: bool = True,
    detect_change: bool = True
):
    dataset1 = Dataset(data_source1, get_color = False, apply_pose = False)
    dataset2 = Dataset(data_source2, get_color = False, apply_pose = False)

    if vdb:
        vdbpipeline1 = VDBPipeline(dataset1, config, jump, n_scans, f"mickie_scans_{str(n_scans)}")
        vdbpipeline1.run()
        vdbpipeline1.draw_mesh() if visualize else None

        vdbpipeline2 = VDBPipeline(dataset2, config, jump, n_scans, f"mickie_scans_{str(n_scans)}")
        vdbpipeline2.run()
        vdbpipeline2.draw_mesh() if visualize else None

    else:
        pipeline = TSDFPipeline(dataset1, config, jump, n_scans, f"mickie_scans_{str(n_scans)}")
        pipeline.run()
        pipeline.draw_mesh() if visualize else None
        
    #changepipeline = ChangeDetectionPipeline(pipeline1._res["mesh"], pipeline2._res["mesh"], config)
    #changepipeline.run()
    #changepipeline.draw_geometries() if visualize else None
    


if __name__ == "__main__":
    argh.dispatch_command(main)
