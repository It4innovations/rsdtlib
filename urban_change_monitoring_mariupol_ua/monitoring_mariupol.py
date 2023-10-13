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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import time
import math
import datetime
import numpy as np
from datetime import datetime
import sys
sys.path.append('../lib/')
import rsdtlib
import multiprocessing
import calendar
from os import walk
import re

#n_threads = 8
n_threads = 4

tf_record_path = "/work/Mariupol/tf_stack_monitoring/"
infer_out_path = "/work/Mariupol/infer_monitoring/"
if not os.path.isdir(infer_out_path):
    os.mkdir(infer_out_path)
infer_1 = infer_out_path + "1/"
if not os.path.isdir(infer_1):
    os.mkdir(infer_1)
infer_2 = infer_out_path + "2/"
if not os.path.isdir(infer_2):
    os.mkdir(infer_2)
infer_3 = infer_out_path + "3/"
if not os.path.isdir(infer_3):
    os.mkdir(infer_3)
infer_4 = infer_out_path + "4/"
if not os.path.isdir(infer_4):
    os.mkdir(infer_4)

model_path = "/work/ukraine/"

# These pre-trained models are not contained here but can be found at:
# https://github.com/It4innovations/urban_change_monitoring_mariupol_ua
best_weights_file1 = "../V1_transfer_116.h5"
best_weights_file2 = "../V2_transfer_113.h5"
best_weights_file3 = "../V3_transfer_146.h5"
best_weights_file4 = "../V4_transfer_109.h5"

# Define the window parameters.
# Note: This is not yet processing!
window = rsdtlib.Window(
                  tf_record_path,
                  60*60*24*182,          # Delta (size)
                  1,                     # window shift
                  35,                    # omega (min. window size)
                  math.ceil(182/2) + 1,  # Omega (max. window size)
                  False,                 # generate triplet
                  n_threads = 1)         # number of threads to use

def infer_it(args):
    import tensorflow as tf
    sys.path.append(model_path)
    from model import ERCNN_DRS

    tile = args[0]
    start_idx = args[1]
    end_idx = args[2]
    to_take = end_idx - start_idx + 1

    windows_ds = window.get_infer_dataset(tile)

    # Do inference here and write result
    ercnn_drs = ERCNN_DRS(True, True)
    model1 = ercnn_drs.build_model(92, 93, 93, 17)
    model1.load_weights(best_weights_file1)
    model2 = ercnn_drs.build_model(92, 93, 93, 17)
    model2.load_weights(best_weights_file2)
    model3 = ercnn_drs.build_model(92, 93, 93, 17)
    model3.load_weights(best_weights_file3)
    model4 = ercnn_drs.build_model(92, 93, 93, 17)
    model4.load_weights(best_weights_file4)

    final_ds = windows_ds.map(lambda x: tf.concat(
                                        [x[1],
                                         x[2],
                                         x[3]],
                                        axis=-1)).skip(start_idx).take(to_take)
    final_ds = final_ds.padded_batch(
                            90, # batch size (requires ~200 GB memory)
                            padded_shapes=([92, 93, 93, 17]),
                            padding_values=-1.0)

    if start_idx == 0:
        result1 = tf.zeros([0, 93, 93])
        result2 = tf.zeros([0, 93, 93])
        result3 = tf.zeros([0, 93, 93])
        result4 = tf.zeros([0, 93, 93])
    else:
        result1 = np.load(
                    infer_1 + "{}_{}.npy".format(tile[0], tile[1]))[:start_idx]
        result2 = np.load(
                    infer_2 + "{}_{}.npy".format(tile[0], tile[1]))[:start_idx]
        result3 = np.load(
                    infer_3 + "{}_{}.npy".format(tile[0], tile[1]))[:start_idx]
        result4 = np.load(
                    infer_4 + "{}_{}.npy".format(tile[0], tile[1]))[:start_idx]

    for item in final_ds:
        result1 = tf.concat([result1, model1(item)], axis=0)
        result2 = tf.concat([result2, model2(item)], axis=0)
        result3 = tf.concat([result3, model3(item)], axis=0)
        result4 = tf.concat([result4, model4(item)], axis=0)

    np.save(infer_1 + "{}_{}.npy".format(tile[0], tile[1]), result1)
    np.save(infer_2 + "{}_{}.npy".format(tile[0], tile[1]), result2)
    np.save(infer_3 + "{}_{}.npy".format(tile[0], tile[1]), result3)
    np.save(infer_4 + "{}_{}.npy".format(tile[0], tile[1]), result4)
    return

if __name__ == '__main__':
    csv_file_name = infer_out_path + "windows_inference.csv"

    window_list = window.windows_list()

    # Discard incomplete windows at end
    last_time = None
    for idx, item in reversed(list(enumerate(window_list))):
        if last_time == None:
            last_time = item[2]
            idx_of_end = idx
            continue

        idx_of_end = idx
        if last_time != item[2]:
            break

    window_list = window_list[0:idx_of_end+1]
    print("Last index of series: {}".format(idx_of_end))

    if os.path.isfile(csv_file_name):
        old_window_list = []
        with open(csv_file_name, mode = "r") as csv_file:
            csv_reader = csv.reader(csv_file,
                                    delimiter=",",
                                    quotechar="\"")
            for row in csv_reader:
                old_window_list.append(
                        (int(row[0]),
                         int(calendar.timegm(
                                 datetime.strptime(
                                        row[1],
                                        '%Y-%m-%d %H:%M:%S').timetuple())),
                         int(calendar.timegm(
                                 datetime.strptime(
                                        row[2],
                                        '%Y-%m-%d %H:%M:%S').timetuple())),
                         int(row[3])))

        if window_list[-1][1] != old_window_list[-1][1]:
            idx_to_start_from = old_window_list[-1][0] + 1
            print("Existing windows present, starting from index: {}".format(
                                                            idx_to_start_from))
        else:
            print("New window list is identical to old one. Aborting")
            sys.exit(1)
    else:
        idx_to_start_from = 0

    if idx_to_start_from > idx_of_end:
        print("Inconsistency of lists. Aborting")
        sys.exit(1)

    ex_files = next(walk(infer_1), (None, None, []))[2]
    ex_tiles = [(int(re.split(r'_|\.', f)[0]), int(re.split(r'_|\.', f)[1])) for f in ex_files if f.endswith('.npy')]

    # Iterate over the tiles and run inference on each concurrently.
    list_args = []

    selector = lambda j, i: True # create all tiles
    #selector = lambda j, i: (j, i) not in ex_tiles # only non-existing tiles

    num_tiles_y, num_tiles_x = window.get_num_tiles()

    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            if selector(j, i):
                list_args.append(((j, i), idx_to_start_from, idx_of_end))

    with multiprocessing.get_context("spawn").Pool(processes = n_threads) as p:
        for i, _ in enumerate(p.imap_unordered(
                                        infer_it,
                                        list_args)):
            sys.stdout.write('\r  Progress: {0:.1%}'.format(i/len(list_args)))
            sys.stdout.flush()
    print('\n')

    # Write the identified windows to a CSV files.
    with open(infer_out_path + "windows_inference.csv",
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
