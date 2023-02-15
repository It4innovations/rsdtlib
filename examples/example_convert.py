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
import sys
sys.path.append("../lib/")
import rsdtlib

# Just a small area in Ostrava/CZ
my_aoi = "./ostrava.shp"

# Locations where to store the EOPatches
dst_path = "./obs/LS5/"
if not os.path.isdir(dst_path):
    os.makedirs(dst_path, exist_ok=True)

# Convert observations (as GeoTIFF files) to EOPatches to be stored in
# 'dst_path' and restricted to AoI 'my_aoi'.
# Note: This does not yet start processing!
convert = rsdtlib.Convert(
                        dst_path=dst_path,
                        aoi=my_aoi)

# Example of converting one GeoTIFF pair (one file with all bands, and one with
# mask).
sample = convert.process(
                        root_path="./LS5/",
                        bands_tiff="19931010T090051.TIF",   # LS5 TM 7 bands
                        mask_tiff="19931010T090051_QA.TIF", # LS5 TM QA band
                        timestamp="19931010T090051")

# Shows the EOPatch:
# EOPatch(
#   data={
#     Bands: numpy.ndarray(shape=(1, 46, 87, 7), dtype=float32)
#   }
#   mask={
#     Mask: numpy.ndarray(shape=(1, 46, 87, 1), dtype=uint16)
#   }
#   bbox=BBox(((18.14081078, 49.83455151),
#              (18.17269592, 49.85132019)), crs=CRS('4326'))
#   timestamp=[datetime.datetime(1993, 10, 10, 9, 0, 51)]
# )
print(sample)
