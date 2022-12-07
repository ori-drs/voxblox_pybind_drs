#!/usr/bin/env python3
# @file      mickie_pipeline.py
#
# Copyright (c) 2021 Ignacio Vizzo, all rights reserved
import argh

from mickie import MickieDataset as Dataset
from tsdf_pipeline import TSDFPipeline
from vdbf_pipeline import VDBPipeline


def main(
    data_source: str, 
    config: str = "/home/joe/git/voxblox_pybind_drs/examples/python/config/mickie.yaml",
    #config: str = "config/mickie.yaml",
    n_scans: int = -1,
    jump: int = 1,
    visualize: bool = True,
):
    """Help here!"""
    dataset = Dataset(data_source, get_color = False, apply_pose = False)
    pipeline = TSDFPipeline(dataset, config, jump, n_scans, f"mickie_scans_{str(n_scans)}")

    vdbpipeline = VDBPipeline(dataset, config, jump, n_scans, f"mickie_scans_{str(n_scans)}")
    vdbpipeline.run()
    #pipeline.run()
    #pipeline.draw_mesh() if visualize else None
    


if __name__ == "__main__":
    argh.dispatch_command(main)
