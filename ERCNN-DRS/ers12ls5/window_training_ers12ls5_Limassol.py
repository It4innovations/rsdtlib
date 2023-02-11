#
# Author: Georg Zitzlsberger (georg.zitzlsberger<ad>vsb.cz)
# Copyright (C) 2020-2023 Georg Zitzlsberger, IT4Innovations,
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
sys.path.append("../../lib/")
import rsdtlib
import multiprocessing

n_threads = 4

dst_path = "./Limassol/"
tf_record_path = "{}/tf_stack/".format(dst_path)
tf_record_out_path = "{}/tf_window/".format(dst_path)
if not os.path.isdir(tf_record_out_path):
    os.mkdir(tf_record_out_path)

tile_size_x = 32
tile_size_y = 32

# Define the window parameters.
# Note: This is not yet processing!
window = rsdtlib.Window(
                  tf_record_path,
                  tf_record_out_path,
                  60*60*24*365,          # Delta (size)
                  1,                     # window stride
                  25,                    # omega (min. window size)
                  110,                   # Omega (max. window size)
                  tile_size_x,           # tile size x
                  tile_size_y,           # tile size y
                  1,                     # bands SAR
                  7,                     # bands opt
                  True,                  # generate triplet
                  n_threads=n_threads,   # number of threads to use
                  use_new_save=False)    # new TF Dataset save

def write_it(args):
    import tensorflow as tf
    sys.path.append('./label/')
    from label import Synthetic_Label

    location = args[0]
    tile = args[1]

    alpha = 0.50

    betas_file = tf_record_out_path + "betas.npy"
    if os.path.exists(betas_file):
        window_betas = np.load(betas_file)
    else:
        assert False, "The betas file has to be present to continue!"

    label_args_ds = tf.data.Dataset.from_tensor_slices(window_betas)

    gen_label = lambda data, label_args:                                       \
                  (tf.concat(
                        # Only serialize current window (index 1)
                        [data[1][1][:, :, :, :],  # SAR ascending
                         data[1][2][:, :, :, :],  # SAR descending
                         data[1][3][:, :, :, :]], # Optical
                         axis=-1),
                   tf.ensure_shape(tf.numpy_function(
                        Synthetic_Label.compute_label_LS5_ERS12_ENDISI,
                        [data[1][1][:, :, :, :], # SAR ascending
                         data[1][2][:, :, :, :], # SAR descending
                         data[1][3][:, :, :, :], # Optical
                         data[0][3][:, :, :, :], # Optical (prev)
                         data[2][3][:, :, :, :], # Optical (next)
                         alpha,
                         label_args[0],
                         label_args[1]], tf.float32),
                        [tile_size_y, tile_size_x]))

    window.write_tf_files(location,
                          lambda j, i: (j==tile[0] and i==tile[1]),
                          label_args_ds=label_args_ds,
                          gen_label=gen_label)
    return


def betas_preproc():
    import numpy as np
    import sys
    import tensorflow as tf
    sys.path.append('./label/')
    from label import Synthetic_Label

    def generate_beta_coeffs(stack, prev_win, next_win):
        import tensorflow as tf

        return tf.ensure_shape(tf.numpy_function(
                    Synthetic_Label.compute_label_LS5_ERS12_ENDISI_beta_coeffs,
                    [prev_win[3], next_win[3]], tf.float32),
                    [2, 3])

    # Helper for parallel execution...
    def get_coeffs(tile):
        print("{}, {}".format(tile[0], tile[1]))

        windows_ds = window.get_infer_dataset(tile)
        coeffs_ds = windows_ds.map(generate_beta_coeffs,
                                   num_parallel_calls=1)

        res = []
        for coeff in coeffs_ds:
            res.append(coeff)
        return np.array(res)


    num_tiles_y, num_tiles_x = window.get_num_tiles()

    print("Computing beta coefficients for tiles (y, x):")
    list_tiles = []
    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            list_tiles.append((j, i))

    co_ds = tf.data.Dataset.from_generator(lambda: list_tiles, tf.uint64)
    co_ds = co_ds.map(lambda tile: tf.py_function(
                                    get_coeffs,
                                    [tile],
                                    [tf.float32]),
                                    num_parallel_calls=n_threads)

    # Attention: Order of tiles is not guaranteed due to parallelism!
    all_coeffs = []
    for item in co_ds:
        all_coeffs.append(item[0])

    # Layout of all_coeffs:
    # [num_tiles_y*num_tiles_x][win#][2][3]
    # - 1st: tiles in flat sequence
    # - 2nd: window number
    # - 3rd: prev_win/next_win
    # - 4th: three coefficients
    all_coeffs = np.array(all_coeffs)

    # Merge all coefficients to buld final beta values for every window
    # Note: A beta value is across all tiles!
    print("Computing beta values for windows:")
    window_betas_tmp = []
    for window_no in range(0, all_coeffs.shape[1]):
        print(window_no)
        beta1 = Synthetic_Label.compute_label_LS5_ERS12_ENDISI_comp_betas(
                                          all_coeffs[:, window_no, 0, :])
        beta2 = Synthetic_Label.compute_label_LS5_ERS12_ENDISI_comp_betas(
                                          all_coeffs[:, window_no, 1, :])
        window_betas_tmp.append((beta1, beta2))
    return np.array(window_betas_tmp)


if __name__ == '__main__':
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

    # Create the beta values for each window (needed for synthetic labeling).
    # Also save them so they don't need to be recomputed.
    betas_file = tf_record_out_path + "betas.npy"
    if not os.path.exists(betas_file):
        window_betas = betas_preproc()
        np.save(betas_file, window_betas)

    # Write the final training samples (windows with labels). The  selector
    # function specifies the tiles to consider for training samples.
    list_tiles = []
    selector = lambda j, i: (j + i/2) % 2 == 0 and                             \
                            not(j >= 15 and i >= 45 and i>=48+45-j)
    num_tiles_y, num_tiles_x = window.get_num_tiles()
    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            if selector(j, i):
                list_tiles.append(("./train/", (j, i)))

    with multiprocessing.get_context("spawn").Pool(processes=n_threads) as p:
        for i, _ in enumerate(p.imap_unordered(
                                        write_it,
                                        list_tiles)):
            sys.stdout.write('\r  Progress: {0:.1%}'.format(i/len(list_tiles)))
            sys.stdout.flush()
    print('\n')

    # Write the final validation samples (windows with labels). The selector
    # function specifies the tiles to consider for validation samples.
    list_tiles = []
    selector = lambda j, i: (j + (i+1)/2 + 1) % 4 == 0 and                     \
                            not(j >= 15 and i >= 45 and i>=48+45-j)
    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            if selector(j, i):
                list_tiles.append(("./val/", (j, i)))

    with multiprocessing.get_context("spawn").Pool(processes=n_threads) as p:
        for i, _ in enumerate(p.imap_unordered(
                                        write_it,
                                        list_tiles)):
            sys.stdout.write('\r  Progress: {0:.1%}'.format(i/len(list_tiles)))
            sys.stdout.flush()
    print('\n')
