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
from dateutil.parser import isoparse
from sentinelhub import SHConfig
from sentinelhub import DataCollection
import sys
sys.path.append('../lib/')
import rsdtlib

# Credentials to access Sentinel Hub. See instructions:
# https://github.com/sentinel-hub/eo-learn/blob/master/examples/README.md
shconfig = SHConfig()
shconfig.instance_id = "<YOUR INSTANCE ID>"
shconfig.sh_client_id = "<YOUR CLIENT ID>"
shconfig.sh_client_secret = "<YOUR CLIENT SECRET>"

# Just a small area in Ostrava/CZ
my_aoi = "./ostrava.shp"

# Locations where to store the EOPatches
dst_path = "./obs"
dst_s1_asc = "{}/S1_asc".format(dst_path)
dst_s1_dsc = "{}/S1_dsc".format(dst_path)
dst_s2 = "{}/S2".format(dst_path)

# Retrieve the observations within the given period
# Note: This does not yet start the download!
retrieve = rsdtlib.Retrieve(
                        starttime = isoparse('20170101T000000'),
                        endtime = isoparse('20170701T000000'),
                        aoi = my_aoi,
                        shconfig = shconfig)

# Start download for Sentinel 1 ascending orbit direction (SAR)
if not os.path.isdir(dst_s1_asc):
    os.makedirs(dst_s1_asc, exist_ok=True)
num_down = retrieve.get_images(datacollection=DataCollection.SENTINEL1_IW_ASC,
                               dst_path=dst_s1_asc)
print("Downloaded to {}: {}".format(dst_s1_asc, num_down))

# Start download for Sentinel 1 descending orbit direction (SAR)
if not os.path.isdir(dst_s1_dsc):
    os.makedirs(dst_s1_dsc, exist_ok=True)
num_down = retrieve.get_images(datacollection=DataCollection.SENTINEL1_IW_DES,
                               dst_path=dst_s1_dsc)
print("Downloaded to {}: {}".format(dst_s1_dsc, num_down))

# Start download for Sentinel 2 (optical)
if not os.path.isdir(dst_s2):
    os.makedirs(dst_s2, exist_ok=True)
num_down = retrieve.get_images(datacollection=DataCollection.SENTINEL2_L1C,
                               dst_path=dst_s2)
print("Downloaded to {}: {}".format(dst_s2, num_down))
