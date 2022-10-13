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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import time
import datetime
import numpy as np
from datetime import datetime
import sys
sys.path.append('../lib/')
import rsdtlib

tf_record_path = "./tf_stack/"
tf_record_out_path = "./tf_window/"
if not os.path.isdir(tf_record_out_path):
    os.mkdir(tf_record_out_path)

# Define the window parameters.
# Note: This is not yet processing!
window = rsdtlib.Window(
                  tf_record_path,
                  tf_record_out_path,
                  60*60*24*30,  # Delta (size)
                  60*60*24*2,   # delta (step)
                  1,            # window shift
                  10,           # omega (min. window size)
                  32,           # tile size x
                  32,           # tile size y
                  13,           # bands opt
                  2,            # bands SAR
                  False)        # generate labels


# Write the identified windows to a CSV files.
window_list = window.windows_list()
with open(tf_record_out_path + "windows_inference.csv", mode = "w") as csv_file:
    csv_writer = csv.writer(csv_file,
                            delimiter=",",
                            quotechar="\"",
                            quoting=csv.QUOTE_MINIMAL)
    for item in window_list:
        csv_writer.writerow([item[0],
                             datetime.utcfromtimestamp(item[1]),
                             datetime.utcfromtimestamp(item[2]),
                             item[3]])

# Create the beta values for each window (needed for synthetic labeling). Also
# save them so they don't need to be recomputed.
not_used = window.preproc()
assert not_used == None, "Preprocessing is not meaningful for inference"

# Write the final inference samples (windows without labels). The lambda
# function specifies the tiles to consider for inference samples.
window.write_tf_files("./infer/",
                      lambda j, i: True)
