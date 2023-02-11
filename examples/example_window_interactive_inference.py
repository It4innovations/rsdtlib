#
# Author: Georg Zitzlsberger (georg.zitzlsberger<ad>vsb.cz)
# Copyright (C) 2020-2022 Georg Zitzlsberger, IT4Innovations,
#                         VSB-Technical University of Ostrava, Czech Republic
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import csv
import time
import math
import datetime
import numpy as np
from datetime import datetime
import sys
sys.path.append("../lib/")
import rsdtlib
import multiprocessing

n_threads = 2

tf_record_path = "./tf_stack/"
infer_out_path = "./infer_results/"
if not os.path.isdir(infer_out_path):
    os.mkdir(infer_out_path)

# Define the window parameters.
# Note: This is not yet processing!
window = rsdtlib.Window(
                  tf_record_path,
                  None,                  # No need for "tf_record_out_path"
                  60*60*24*30,           # Delta (size)
                  1,                     # window stride
                  10,                    # omega (min. window size)
                  16,                    # Omega (max. window size)
                  32,                    # tile size x
                  32,                    # tile size y
                  2,                     # bands SAR
                  13,                    # bands opt
                  False,                 # generate triplet
                  n_threads=n_threads)   # number of threads to use

def infer_it(tile):
    import tensorflow as tf

    windows_ds = window.get_infer_dataset(tile)

    # <PREDICT WITH MODEL>
    # E.g., result = model.predict(final_ds)

    return

if __name__ == "__main__":
    # Verbose identified windows.
    window_list = window.windows_list()

    # Iterate over the tiles and run inference on each concurrently.
    list_tiles = []
    selector = lambda j, i: True # Use all tiles
    num_tiles_y, num_tiles_x = window.get_num_tiles()
    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            if selector(j, i):
                list_tiles.append((j, i))

    with multiprocessing.get_context("spawn").Pool(processes=n_threads) as p:
        for i, _ in enumerate(p.imap_unordered(
                                        infer_it,
                                        list_tiles)):
            sys.stdout.write("\r  Progress: {0:.1%}".format(i/len(list_tiles)))
            sys.stdout.flush()
    print("\n")
