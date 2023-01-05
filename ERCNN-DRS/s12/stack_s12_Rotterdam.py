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
from datetime import datetime
import sys
sys.path.append('../../lib/')
import rsdtlib

# Locations where to find the input observations and where to store the stacked,
# assembled and tiled time series.
dst_path = "./Rotterdam/"
dst_s1_asc = "{}/S1_asc/eopatches/".format(dst_path)
dst_s1_dsc = "{}/S1_dsc/eopatches/".format(dst_path)
dst_s2 = "{}/S2/eopatches/".format(dst_path)
tf_record_path = "{}/tf_stack/".format(dst_path)
if not os.path.isdir(tf_record_path):
    os.mkdir(tf_record_path)

# Stack, assemble and tile the observations.
stack = rsdtlib.Stack(
                dst_s1_asc,
                dst_s1_dsc,
                dst_s2,
                "L1_GND",
                "dataMask",
                "L1C_data",
                "dataMask",
                tf_record_path,
                datetime(2017, 1, 1, 0, 0, 0),
                datetime(2021, 1, 17, 0, 0, 0),
                60*60*24*2,   # delta (step)
                32,           # tile size x
                32,           # tile size y
                13,           # bands opt
                2)            # bands SAR

# Process the stacking, assembling and tiling. Each output (TFRecord) file is
# for one tile.
stack.process()
