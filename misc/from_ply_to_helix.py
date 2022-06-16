import argparse

import numpy as np
from plyfile import PlyData

parser = argparse.ArgumentParser(description="From xxx.ply to helix_xxx.ply")
parser.add_argument("--filename", type=str)
parser.add_argument("--time_scale", default=20., type=float)
args = parser.parse_args()

ply = PlyData.read(args.filename)

mini, maxi = np.percentile(ply["vertex"]["time"], 1), np.percentile(ply["vertex"]["time"], 99)
time = (ply["vertex"]["time"] - mini)/(maxi - mini)

ply["vertex"]["z"] += args.time_scale*ply["vertex"]["frame"].max() * time

ply.write(f"{args.filename[:-4]}_helix.ply")