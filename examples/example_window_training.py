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
tf_record_out_path = "./tf_window/"
os.mkdirs(tf_record_out_path, exist_ok=True)
tf_record_out_path_train = "./tf_window/train/"
os.mkdirs(tf_record_out_path_train, exist_ok=True)
tf_record_out_path_val = "./tf_window/val/"
os.mkdirs(tf_record_out_path_val, exist_ok=True)

# Define the window parameters.
# Note: This is not yet processing!
window = rsdtlib.Window(
                  tf_record_path,
                  60*60*24*30,           # Delta (size)
                  1,                     # window stride
                  10,                    # omega (min. window size)
                  16,                    # Omega (max. window size)
                  False,                 # generate triplet
                  n_threads=n_threads,   # number of threads to use
                  use_new_save=False)    # new TF Dataset save

def write_it(args):
    import tensorflow as tf

    location = args[0]
    tile = args[1]

    # Just some dummy arguments (not used in this example)
    label_args_ds = tf.data.Dataset.from_tensor_slices(
                                                tf.constant([0])).repeat()

    gen_label = lambda window, label_args:                                     \
                  (tf.concat(
                        # Only serialize current window (index 1)
                        [window[0][1][:, :, :, :],  # SAR ascending
                         window[0][2][:, :, :, :],  # SAR descending
                         window[0][3][:, :, :, :]], # optical
                        axis=-1),
                   tf.ensure_shape(                 # label
                        tf.ones([32, 32]), [32, 32]))

    window.write_tf_files(location,
                          lambda j, i: (j==tile[0] and i==tile[1]),
                          label_args_ds=label_args_ds,
                          gen_label=gen_label)
    return

if __name__ == "__main__":
    # Write the identified windows to a CSV files.
    window_list = window.windows_list()
    with open(tf_record_out_path + "windows_training.csv",
              mode = "w") as csv_file:
        csv_writer = csv.writer(csv_file,
                                delimiter=",",
                                quotechar="\"",
                                quoting=csv.QUOTE_MINIMAL)
        for item in window_list:
            csv_writer.writerow([item[0],
                                 datetime.utcfromtimestamp(item[1]),
                                 datetime.utcfromtimestamp(item[2]),
                                 item[3]])

    # Write the final training samples (windows with labels). The  selector
    # function specifies the tiles to consider for training samples.
    list_tiles = []
    selector = lambda j, i: (j + i/2) % 2 == 0
    num_tiles_y, num_tiles_x = window.get_num_tiles()
    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            if selector(j, i):
                list_tiles.append((tf_record_out_path_train, (j, i)))

    print("Writing training files:")
    with multiprocessing.get_context("spawn").Pool(processes=n_threads) as p:
        for i, _ in enumerate(p.imap_unordered(
                                        write_it,
                                        list_tiles)):
            sys.stdout.write("\r  Progress: {0:.1%}".format(i/len(list_tiles)))
            sys.stdout.flush()
    print("\n")

    # Write the final validation samples (windows with labels). The selector
    # function specifies the tiles to consider for validation samples.
    list_tiles = []
    selector = lambda j, i: (j + (i+1)/2 + 1) % 4 == 0
    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            if selector(j, i):
                list_tiles.append((tf_record_out_path_val, (j, i)))

    print("Writing validation files:")
    with multiprocessing.get_context("spawn").Pool(processes=n_threads) as p:
        for i, _ in enumerate(p.imap_unordered(
                                        write_it,
                                        list_tiles)):
            sys.stdout.write("\r  Progress: {0:.1%}".format(i/len(list_tiles)))
            sys.stdout.flush()
    print("\n")
