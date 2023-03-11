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

class Retrieve:
    """
    Class for downloading observations from `Sentinel Hub`
    (https://www.sentinel-hub.com/).

    :param starttime: Starting time of the time series to retrieve
    :type starttime: datetime.datetime
    :param endtime: End time of the time series to retrieve
    :type endtime: datetime.datetime
    :param aoi: Area of Interest (path to shape file)
    :type aoi: str

    **Example:**

    Define a download of observations in the time frame 01-01-2017 to 01-07-2017
    from `Sentinel Hub` for the Area of Interest (AoI) defined in the shape file
    ``ostrava.shp``. An account on Sentinel Hub is required for which the
    ``shconfig`` provides the access credentials.

    .. code-block:: python

        import rsdtlib
        from dateutil.parser import isoparse
        from sentinelhub import SHConfig

        # Credentials to access Sentinel Hub. See instructions:
        # https://github.com/sentinel-hub/eo-learn/blob/master/examples/README.md
        shconfig = SHConfig()
        shconfig.instance_id = "<YOUR INSTANCE ID>"
        shconfig.sh_client_id = "<YOUR CLIENT ID>"
        shconfig.sh_client_secret = "<YOUR CLIENT SECRET>"

        # Just a small area in Ostrava/CZ
        my_aoi = "./ostrava.shp"

        retrieve = rsdtlib.Retrieve(
                        starttime=isoparse("20170101T000000"),
                        endtime=isoparse("20170701T000000"),
                        aoi=my_aoi,
                        shconfig=shconfig)

    """
    def __init__(self, starttime, endtime, aoi, shconfig):
        self.starttime = starttime
        self.endtime = endtime
        self.aoi = aoi
        self.shconfig = shconfig


    def _get_intervall(self):
        """
        Returns monthly intervals to the provided time range:
            :param starttime: - :param endtime:
        """
        from dateutil.relativedelta import relativedelta
        interval_list = []
        this_time = self.starttime
        while this_time < self.endtime:
            this_endtime = this_time + relativedelta(months=+1, seconds=-1)
            if this_endtime > self.endtime:
                this_endtime= self.endtime

            interval_list.append((this_time.isoformat(),
                                  this_endtime.isoformat()))
            this_time += relativedelta(months=+1)
        return interval_list


    def _get_bbox(self):
        """
        Return the bounding box defined by the Area of Interest

        Note: Only the first polygon in the shapefile will be used!
        """
        import geopandas as gpd
        from sentinelhub import geometry

        shp_data = gpd.read_file(self.aoi)
        crs = shp_data.crs.to_epsg()

        for item, row in shp_data.iterrows():
            return geometry.Geometry(row['geometry'], crs=crs).bbox


    def _merge_CLM(self,
                   eopatches_tmp_dir,
                   eopatches_clm_dir,
                   eopatches_out_dir,
                   eopatches_fail_dir):
        """
        Create the final EOPatchs by merging the cloud mask and
        observations. Empty (i.e., fully masked) observations are
        moved to :param eopatches_fail_dir:.
        """
        import os
        import numpy as np
        from eolearn.core import EONode, LoadTask, SaveTask, ZipFeatureTask,   \
                                 EOExecutor, EOWorkflow, EOTask
        from eolearn.core.core_tasks import RemoveFeatureTask
        from eolearn.core import FeatureType

        eopatches_list_all = [f for f in sorted(os.listdir(eopatches_tmp_dir))]
        eopatches_CLM_list_all = [f for f in sorted(os.listdir(
                                                           eopatches_clm_dir))]
        assert eopatches_list_all == eopatches_CLM_list_all,                   \
               "Lists are not identical"

        class MyMergeEOPatchesTask(EOTask):
            def __init__(self, **merge_kwargs):
                self.merge_kwargs = merge_kwargs

            def execute(self, *eopatches):
                if not eopatches:
                    raise ValueError("At least one EOPatch should be given")

                eopatches[0].mask["dataMask"] &= eopatches[1].mask["CLM"] != 1.0
                return eopatches[0]

        class SelectTask(EOTask):
            def __init__(self, passed_dir, failed_dir):
                self.passed_dir = passed_dir
                self.failed_dir = failed_dir

            def execute(self, eopatch, *, eopatch_folder=""):
                if not np.all(eopatch.mask["dataMask"] == False):
                    save_task = SaveTask(self.passed_dir)
                else:
                    save_task = SaveTask(self.failed_dir)

                return save_task.execute(eopatch, eopatch_folder=eopatch_folder)

        load_task = LoadTask(eopatches_tmp_dir)
        load_task_CLM = LoadTask(eopatches_clm_dir)

        merge_eopatches_task = MyMergeEOPatchesTask(
                                    time_dependent_op="concatenate")

        merge_task = ZipFeatureTask({FeatureType.MASK: ["CLM", "dataMask"]},
                                    (FeatureType.MASK, "IS_VALID"),
                                    lambda *f: ~f[0] | f[1])

        remove_task = RemoveFeatureTask({FeatureType.MASK: ["CLM", "dataMask"]})

        save_task = SelectTask(eopatches_out_dir, eopatches_fail_dir)

        node_load = EONode(load_task, inputs=[])
        node_load_CLM = EONode(load_task_CLM, inputs=[])
        node_merge = EONode(merge_eopatches_task,
                            inputs=[node_load, node_load_CLM])
        node_save = EONode(save_task, inputs=[node_merge])
        workflow = EOWorkflow([node_load,
                               node_load_CLM,
                               node_merge,
                               node_save])

        exec_args = []
        for eopatch in eopatches_list_all:
            if (os.path.isdir(eopatches_out_dir + eopatch) or
                os.path.isdir(eopatches_fail_dir + eopatch)):
                continue

            exec_args.append({
                node_load: {
                    "eopatch_folder": eopatch
                },
                node_load_CLM: {
                    "eopatch_folder": eopatch
                },
                node_merge: {
                },
                node_save: {
                    "eopatch_folder": eopatch
                }
            })

        coexec = EOExecutor(workflow, exec_args)
        coexec.run(workers=1)


    def _filter_empty_SAR(self,
                          eopatches_tmp_dir,
                          eopatches_out_dir,
                          eopatches_fail_dir):
        """
        Empty (i.e., fully masked) observations are moved to
        :param eopatches_fail_dir:.
        """
        import os
        import shutil
        import numpy as np
        from eolearn.core import EOPatch

        def get_eopatch():
            eopatches_list_all = [f for f in sorted(os.listdir(
                                                          eopatches_tmp_dir))]
            for eop in eopatches_list_all:
                yield eop

        for obs in get_eopatch():
            test_obs = EOPatch.load("{}/{}".format(eopatches_tmp_dir, obs))
            if np.all(test_obs.mask["dataMask"] == False):
                print("Found empty: {}".format(obs))
                shutil.move(
                        "{}/{}".format(eopatches_tmp_dir, obs),
                        eopatches_fail_dir)
            else:
                shutil.move(
                        "{}/{}".format(eopatches_tmp_dir, obs),
                        eopatches_out_dir)


    def get_images(self, datacollection, dst_path, maxcc=1.0):
        """
        Download the observations for the specified remote sensing type
        ``datacollection`` and maximum cloud coverage ``maxcc`` (if applicable).
        The observations are stored on the filesystem at ``dst_path``.

        :param datacollection: Data collection to download
        :type datacollection: sentinelhub.DataCollection_
        :param dst_path: Path on filesystem to store observations at
        :type dst_path: str
        :param maxcc: Maximum cloud coverage (default = ``1.0``)
        :type maxcc: float
        :return: Number of retrieved observations
        :rtype: int

        .. _sentinelhub.DataCollection: https://sentinelhub-py.readthedocs.io/en/latest/examples/data_collections.html

        **Example:**

        Start the download of the AoI and time frame specified in the object
        `retrieve`. `Sentinel Hub`'s data collection ``SENTINEL1_IW_ASC``
        is specified to retrieve Sentinel 1 observations in ascending orbit
        direction. The retrieved observations are stored in path ``dst_s1_asc``.

        .. code-block:: python

            from sentinelhub import DataCollection

            dst_s1_asc = "<PATH FOR OBSERVATIONS>"

            num_down = retrieve.get_images(
                                datacollection=DataCollection.SENTINEL1_IW_ASC,
                                dst_path=dst_s1_asc)

        """
        import os
        import sys
        import shutil
        import datetime
        from sentinelhub import DataCollection
        from eolearn.core import EOPatch, FeatureType, SaveTask
        from eolearn.io.sentinelhub_process import get_available_timestamps
        from eolearn.io import SentinelHubInputTask
        from eolearn.core import EOWorkflow, linearly_connect_tasks

        if not os.path.isdir(dst_path):
            print("Error: Destination directory does not exist: {}".format(
                                                                      dst_path))
            return

        eopatches_tmp_dir = "{}/tmp_eopatches/".format(dst_path)
        if os.path.isdir(eopatches_tmp_dir):
            print("Warning: Temporary directory already exists: {}".format(
                                                             eopatches_tmp_dir))
#            return
        else:
            os.mkdir(eopatches_tmp_dir)

        if datacollection == DataCollection.SENTINEL2_L1C:
            eopatches_clm_dir = "{}/CLM_eopatches/".format(dst_path)
            if os.path.isdir(eopatches_clm_dir):
                print("Warning: Temporary cloud mask directory already " +
                      "exists: {}".format(eopatches_clm_dir))
#                return
            else:
                os.mkdir(eopatches_clm_dir)

        eopatches_out_dir = "{}/eopatches/".format(dst_path)
        if not os.path.isdir(eopatches_out_dir):
            os.mkdir(eopatches_out_dir)

        eopatches_fail_dir = "{}/fail_eopatches/".format(dst_path)
        if not os.path.isdir(eopatches_fail_dir):
            os.mkdir(eopatches_fail_dir)

        time_interval = (self.starttime.isoformat(), self.endtime.isoformat())
        this_bbox = self._get_bbox()

        # Get number of samples to download...
        if datacollection == DataCollection.SENTINEL2_L1C:
            all_samples = get_available_timestamps(
                                this_bbox,
                                config=self.shconfig,
                                data_collection=datacollection,
                                maxcc=maxcc,
                                time_difference=datetime.timedelta(seconds=-1),
                                time_interval=time_interval)
        elif (datacollection == DataCollection.SENTINEL1_IW_ASC or
              datacollection == DataCollection.SENTINEL1_IW_DES):
            all_samples = get_available_timestamps(
                                self._get_bbox(),
                                config=self.shconfig,
                                data_collection=datacollection,
                                maxcc=maxcc,
                                time_difference=datetime.timedelta(seconds=-1),
                                time_interval=time_interval)
        else:
            assert False, "Unkown or unsupported DataCollection"

        # Create tasks for downloading...
        if datacollection == DataCollection.SENTINEL2_L1C:
            download_task = SentinelHubInputTask(
                data_collection=datacollection,
                bands=["B01","B02","B03","B04","B05","B06","B07","B08","B8A",  \
                       "B09","B10","B11","B12"],
                bands_feature=(FeatureType.DATA, "L1C_data"),
                additional_data=[(FeatureType.MASK, "dataMask")],
                resolution=10,
                maxcc=maxcc,
                time_difference=None,
                config=self.shconfig,
                max_threads=4
            )
            save_task = SaveTask(eopatches_tmp_dir)
        elif (datacollection == DataCollection.SENTINEL1_IW_ASC or
              datacollection == DataCollection.SENTINEL1_IW_DES):
            download_task = SentinelHubInputTask(
                data_collection=datacollection,
                bands=["VV","VH"],
                bands_feature=(FeatureType.DATA, "L1_GND"),
                additional_data=[(FeatureType.MASK, "dataMask")],
                resolution=10,
                maxcc=maxcc,
                time_difference=None,
                config=self.shconfig,
                max_threads=4,
                aux_request_args = {"processing":{
                                        "orthorectify":"True",
                                        "demInstance":"COPERNICUS"}}
            )
            save_task = SaveTask(eopatches_tmp_dir)
        else:
            assert False, "Unkown or unsupported DataCollection"


        workflow_nodes = linearly_connect_tasks(download_task, save_task)
        downloader = EOWorkflow(workflow_nodes)

        num_down = 0
        for i in all_samples:
            start_d = i - datetime.timedelta(0,1)
            end_d = i + datetime.timedelta(0,1)
            this_time_str = i.strftime("%Y%m%dT%H%M%S")
            if (os.path.isdir("{}/{}".format(eopatches_out_dir,
                                            this_time_str)) or
                os.path.isdir("{}/{}".format(eopatches_fail_dir,
                                            this_time_str)) or
                os.path.isdir("{}/{}".format(eopatches_tmp_dir,
                                            this_time_str))):
                    continue

            download_result = downloader.execute({
                workflow_nodes[0]: {
                    "bbox": this_bbox,
                    "time_interval": (start_d, end_d)
                },
                workflow_nodes[1]: {
                    "eopatch_folder": this_time_str
                }
            })
            num_down += 1
            sys.stdout.write('\r  Data progress: {0:.1%}'.format(
                                                     num_down/len(all_samples)))
            sys.stdout.flush()
        print("\n")

        # For optical data, download also cloud masks and merge. If that fails,
        # move to failed directory.
        # For SAR data, test if observations are empty and move those to failed
        # directory.
        if datacollection == DataCollection.SENTINEL2_L1C:
            get_clm_task = SentinelHubInputTask(
                data_collection=datacollection,
                bands_feature=None,
                additional_data = [
                    (FeatureType.MASK, "CLM")
                ],
                resolution=10,
                maxcc=maxcc,
                time_difference=None,
                config=self.shconfig,
                max_threads=4
            )
            save_task = SaveTask(eopatches_clm_dir)
            workflow_nodes = linearly_connect_tasks(get_clm_task, save_task)
            downloader = EOWorkflow(workflow_nodes)

            num_down = 0
            for i in all_samples:
                start_d = i - datetime.timedelta(0,1)
                end_d = i + datetime.timedelta(0,1)
                this_time_str = i.strftime("%Y%m%dT%H%M%S")
                if (os.path.isdir("{}/{}".format(eopatches_out_dir,
                                                this_time_str)) or
                    os.path.isdir("{}/{}".format(eopatches_fail_dir,
                                                this_time_str)) or
                    os.path.isdir("{}/{}".format(eopatches_clm_dir,
                                                this_time_str))):
                    continue

                download_result = downloader.execute({
                    workflow_nodes[0]: {
                        "bbox": this_bbox,
                        "time_interval": (start_d, end_d)
                    },
                    workflow_nodes[1]: {
                        "eopatch_folder": this_time_str
                    }
                })
                num_down += 1
                sys.stdout.write('\r  CLM progress: {0:.1%}'.format(
                                                     num_down/len(all_samples)))
                sys.stdout.flush()
            print("\n")

            self._merge_CLM(eopatches_tmp_dir,
                            eopatches_clm_dir,
                            eopatches_out_dir,
                            eopatches_fail_dir)
            shutil.rmtree(eopatches_tmp_dir)
            shutil.rmtree(eopatches_clm_dir)
        elif (datacollection == DataCollection.SENTINEL1_IW_ASC or
              datacollection == DataCollection.SENTINEL1_IW_DES):
            self._filter_empty_SAR(eopatches_tmp_dir,
                                   eopatches_out_dir,
                                   eopatches_fail_dir)
            shutil.rmtree(eopatches_tmp_dir)
        else:
            assert False, "Unkown or unsupported DataCollection"

        return num_down


class Convert:
    """
    Class for converting GeoTIFF files to EOPatches.

    :param dst_path: Path on filesystem to store the converted observations at
    :type dst_path: str
    :param aoi: Area of Interest (path to shape file)
    :type aoi: str
    :param normalize: Divisor to use for normalization
      (e.g., 255.0 for 8 bit unsigned integer types)
    :type normalize: float

    **Example:**

    Define a conversion of observations for the Area of Interest (AoI) defined
    in the shape file ``ostrava.shp``. The converted observations are stored in
    in path ``dst_path``.

    .. code-block:: python

        import rsdtlib

        dst_path = "<PATH FOR OBSERVATIONS>"

        # Just a small area in Ostrava/CZ
        my_aoi = "./ostrava.shp"

        convert = rsdtlib.Convert(
                        dst_path=dst_path,
                        aoi=my_aoi)

    """
    def __init__(self,
                 dst_path,
                 aoi,
                 normalize = 255.0):
        self.dst_path = dst_path
        self.aoi = aoi
        self.normalize = normalize

    def process(self, root_path, bands_tiff, mask_tiff, timestamp):
        """
        Start the conversion process. Merge the GeoTIFF with the bands
        ``bands_tiff`` and the GeoTIFF containing a mask ``mask_tiff``
        from the same directory ``root_path``. Annotate the result with a
        timestamp ``timestmap`` and store it as a single EOPatch.

        :param root_path: Root directory of all GeoTIFF files
        :type root_path: str
        :param bands_tiff: Filename of the GeoTIFF containing the bands
        :type bands_tiff: str
        :param mask_tiff: Filename of the GeoTIFF containing the mask
        :type mask_tiff: str
        :param timestamp: Timestamp of the converted observation
        :type timestamp: datetime.datetime
        :return: EOPatch object
        :rtype: eolearn.core.EOPatch

        **Example:**

        Convert observations as specified in the object ``convert``. This is
        called for every single observation provided as two `GeoTIFF` files from
        the ``root_path``. One contains the contents (``bands_tiff``) and one
        the data mask (``mask_tiff``). A timestamp is assigned via
        ``timestamp``.

        .. code-block:: python

            root_path = "<ROOT OF GEOTIFFS>"

            sample = convert.process(
                        root_path=root_path
                        bands_tiff="19931010T090051.TIF",   # LS5 TM 7 bands
                        mask_tiff="19931010T090051_QA.TIF", # LS5 TM QA band
                        timestamp="19931010T090051")

        """
        import numpy as np
        import eolearn.io
        import geopandas as gpd
        from sentinelhub import geometry
        from datetime import datetime
        from eolearn.core import FeatureType, EOTask, EOPatch
        from eolearn.core import EONode, EOWorkflow, EOExecutor
        from sentinelhub import BBox, CRS

        class EOPatchConvert(EOTask):
            def __init__(self,
                         root_path,
                         bands_tiff,
                         mask_tiff,
                         timestamp,
                         bbox):
                self.root_path = root_path
                self.bands_tiff = bands_tiff
                self.mask_tiff = mask_tiff
                self.timestamp = timestamp
                self.bbox = bbox

            def execute(self):
                execution_args = {}

                importer_task = eolearn.io.local_io.ImportFromTiffTask(
                                        (FeatureType.DATA, "Bands"),
                                        image_dtype = np.float32,
                                        folder = self.root_path)
                importer_mask_task = eolearn.io.local_io.ImportFromTiffTask(
                                        (FeatureType.MASK, "Mask"),
                                        image_dtype = np.uint16,
                                        folder = self.root_path)

                this_patch = EOPatch(bbox = self.bbox)
                this_patch = importer_task.execute(this_patch,
                                                   filename=self.bands_tiff)
                this_patch = importer_mask_task.execute(this_patch,
                                                        filename=self.mask_tiff)

                # Normalize 'Bands'
                bands_mod = this_patch.data["Bands"]
                bands_mod = bands_mod[:,:,:,:] / self.normalize
                this_patch.data["Bands"] = bands_mod

                new_datetime = datetime.strptime(self.timestamp,
                                                 "%Y%m%dT%H%M%S")
                this_patch.timestamp = [new_datetime]
                return this_patch

        shp_data = gpd.read_file(self.aoi)
        crs = shp_data.crs.to_epsg()

        this_bbox = {}
        for item, row in shp_data.iterrows():
            this_bbox = geometry.Geometry(row['geometry'], crs=crs).bbox
            break # Just use first entry

        convert = EOPatchConvert(root_path = root_path,
                                    bands_tiff = bands_tiff,
                                    mask_tiff = mask_tiff,
                                    timestamp = timestamp,
                                    bbox = this_bbox)
        result = convert.execute()
        result.save(self.dst_path + "/{}".format(timestamp))
        return result


class Stack:
    """
    Class for stacking, assembling and tiling. The stacking is independent of
    each observation type. Assembling combines all observations over time, of
    types optical multispectral, and Synthetic Aperture Radar (SAR) in
    ascending and descending orbit directions.

    Note: Even though the ``starttime`` might be later, earlier observations
    are still considered for stacking. That is, earlier observation pixels are
    carried forward if masked at ``starttime``. However, only observation time
    stamps are effectively written, that are within the time frame of
    ``starttime`` and ``endtime``.

    :param sar_asc_path: Directory of SAR observations (ascending)
    :type sar_asc_path: str
    :param sar_dsc_path: Directory of SAR observations (descending)
    :type sar_dsc_path: str
    :param opt_path: Directory of optical multispectral observations
    :type opt_path: str
    :param sar_bands_name: Name of the SAR bands
    :type sar_bands_name: str
    :param sar_mask_name: Name of the SAR mask
    :type sar_mask_name: str
    :param opt_bands_name: Name of the optical multispectral bands
    :type opt_bands_name: str
    :param opt_mask_name: Name of the optical multispectral mask
    :type opt_mask_name: str
    :param tf_record_path: Tensorflow Record path to store results
    :type tf_record_path: str
    :param starttime: Starting time of the time series to consider
    :type starttime: datetime.datetime
    :param endtime: End time of the time series to rconsider
    :type endtime: datetime.datetime
    :param delta_step: The step value (:math:`\delta`) to avoid redundant
        observations temporally close together
    :type delta_step: int
    :param tile_size_x: Tile size in ``x`` dimension
    :type tile_size_x: int
    :param tile_size_y: Tile size in ``y`` dimension
    :type tile_size_y: int
    :param bands_sar: Number of SAR bands for each orbit direction
        (:math:`b_{SAR}^{[asc\mid dsc]}`)
    :type bands_sar: int
    :param bands_opt: Number of optical multispectral bands (:math:`b_{OPT}`)
    :type bands_opt: int
    :param deadzone_x: Overlap tiles on x-axis so that neighboring tiles share
        ``2*deadzone_x`` pixels per side (default = ``0``)
    :type deadzone_x: int
    :param deadzone_y: Overlap tiles on y-axis so that neighboring tiles share
        ``2*deadzone_y`` pixels per side (default = ``0``)
    :type deadzone_y: int

    **Example:**

    Define the stacking, assembling, and tiling with different parameters.
    Observations are taken from ``sar_asc_path``/``sar_dsc_path`` for SAR data,
    and ``opt_path`` for optical multispectral data. The SAR and optical bands
    ("L1_GND" and "L1C_data"), and their masks ("dataMask" each) are
    explicitly specified. The timeframe to consider for stacking is 01-01-2017
    till 01-07-2017. The step value :math:`\delta` is set to two days, and
    ``y`` and ``x`` sizes of the tiles to 32 pixels each. The number of bands
    :math:`b_{SAR}^{[asc\mid dsc]}` is two (for each oribt direction), and
    :math:`b_{OPT}` is 13.

    .. code-block:: python

        import rsdtlib
        from datetime import datetime

        sar_asc_path = "<SOURCE PATH OF S1 ASC OBSERVATIONS>"
        sar_dsc_path = "<SOURCE PATH OF S1 DSC OBSERVATIONS>"
        opt_path = "<SOURCE PATH OF S2 OBSERVATIONS>"
        tf_record_path = "<DESTINATION PATH OF TFRECORD FILES>"

        stack = rsdtlib.Stack(
                        sar_asc_path,
                        sar_dsc_path,
                        opt_path,
                        "L1_GND",
                        "dataMask",
                        "L1C_data",
                        "dataMask",
                        tf_record_path,
                        datetime(2017, 1, 1, 0, 0, 0),
                        datetime(2017, 7, 1, 0, 0, 0),
                        60*60*24*2,   # every two days
                        32,           # tile size x
                        32,           # tile size y
                        2,            # bands SAR
                        13)           # bands opt

    """
    from enum import Enum as _Enum
    class _RS_Type(_Enum):
        UNKOWN = -1
        OPT = 0
        SAR_ASC = 1
        SAR_DSC = 2


    def __init__(self,
                 sar_asc_path,
                 sar_dsc_path,
                 opt_path,
                 sar_bands_name,
                 sar_mask_name,
                 opt_bands_name,
                 opt_mask_name,
                 tf_record_path,
                 starttime,
                 endtime,
                 delta_step,
                 tile_size_x,
                 tile_size_y,
                 bands_sar,
                 bands_opt,
                 deadzone_x = 0,
                 deadzone_y = 0):
        self.sar_asc_path = sar_asc_path
        self.sar_dsc_path = sar_dsc_path
        self.opt_path = opt_path
        self.sar_bands_name = sar_bands_name
        self.sar_mask_name = sar_mask_name
        self.opt_bands_name = opt_bands_name
        self.opt_mask_name = opt_mask_name
        self.tf_record_path = tf_record_path
        self.starttime = starttime
        self.endtime = endtime
        self.delta_step = delta_step
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.bands_sar = bands_sar
        self.bands_opt = bands_opt
        self.deadzone_x = deadzone_x
        self.deadzone_y = deadzone_y


    def _getEOPatches(self, path_root):
        import os
        import dateutil.parser

        all_files = []
        list_patches = [dateutil.parser.parse(name)                            \
                        for name in os.listdir(path_root)                      \
                        if os.path.isdir("{}{}".format(path_root, name))]
        list_patches.sort()

        time_patches = []
        for patch_dt in list_patches:
            new_datetime = patch_dt.strftime("%Y%m%dT%H%M%S")
            if os.path.exists(path_root + new_datetime):
                all_files.append((patch_dt, path_root + new_datetime))
        return all_files


    def _get_min_resolution(self,
                            all_files_OPT,
                            all_files_SAR_ascending,
                            all_files_SAR_descending):
        from eolearn.core import EOPatch

        min_height_OPT = None
        min_width_OPT = None
        min_height_SAR_ascending = None
        min_width_SAR_ascending = None
        min_height_SAR_descending = None
        min_width_SAR_descending = None

        for fle in all_files_OPT:
            ptch = EOPatch.load(fle[1])
            height = ptch.data[self.opt_bands_name].shape[1]
            width = ptch.data[self.opt_bands_name].shape[2]
            if (min_height_OPT is None or min_height_OPT > height):
                min_height_OPT = height
            if (min_width_OPT is None or min_width_OPT > width):
                min_width_OPT = width
            break # OPT has consistent sizes

        for fle in all_files_SAR_ascending:
            ptch = EOPatch.load(fle[1])
            height = ptch.data[self.sar_bands_name].shape[1]
            width = ptch.data[self.sar_bands_name].shape[2]
            if (min_height_SAR_ascending is None or
                min_height_SAR_ascending > height):
                min_height_SAR_ascending = height
            if (min_width_SAR_ascending is None or
                min_width_SAR_ascending > width):
                min_width_SAR_ascending = width
            break # SAR has consistent sizes

        for fle in all_files_SAR_descending:
            ptch = EOPatch.load(fle[1])
            height = ptch.data[self.sar_bands_name].shape[1]
            width = ptch.data[self.sar_bands_name].shape[2]
            if (min_height_SAR_descending is None or
                min_height_SAR_descending > height):
                min_height_SAR_descending = height
            if (min_width_SAR_descending is None or
                min_width_SAR_descending > width):
                min_width_SAR_descending = width
            break # SAR has consistent sizes

        min_height_SAR = min(min_height_SAR_ascending,
                             min_height_SAR_descending)
        min_width_SAR = min(min_width_SAR_ascending,
                            min_width_SAR_descending)
        return min_height_OPT, min_width_OPT, min_height_SAR, min_width_SAR


    def _get_steps_list(self, list_time_stamps):
        from datetime import timedelta

        # Run through once to identify where to set the steps
        start_cur_window = self.starttime
        prev_ts = None
        eff_windows = 0
        steps_list = []
        for ts in list_time_stamps:
            if ts[0] < start_cur_window: # only for the beginning
                continue
            if ts[0] >= self.endtime: # only for past the end date
                break
            if ts[0] - start_cur_window >= timedelta(seconds=self.delta_step):
                eff_windows += 1
                off_by = (ts[0] - start_cur_window)//timedelta(
                                                        seconds=self.delta_step)
                start_cur_window += off_by * timedelta(seconds=self.delta_step)
                if prev_ts != None: # only for the beginning
                    steps_list.append(prev_ts)
            prev_ts = ts[0]
        if prev_ts != None and not prev_ts in steps_list:
            eff_windows += 1
            steps_list.append(prev_ts)

        print("Effective time stamps " +
              "(from 'starttime' with 'delta_step' steps): {}".format(
                                                                eff_windows))
        return steps_list


    def _tile_stream(self,
                     stream,
                     height,
                     width,
                     tile_size_y,
                     tile_size_x,
                     channels):
        import tensorflow as tf

        stream_t = tf.image.extract_patches(
                tf.reshape(stream, [1, height, width, channels]),              \
                           sizes=[1, tile_size_y, tile_size_x, 1],             \
                           strides=[1,
                                    tile_size_y - 2 * self.deadzone_y,
                                    tile_size_x - 2 * self.deadzone_x,
                                    1],                                        \
                           rates=[1, 1, 1, 1],                                 \
                           padding="VALID")
        cor_stream_t = tf.reshape(stream_t,                                    \
                                  [stream_t.shape[1],                          \
                                   stream_t.shape[2],                          \
                                   tile_size_y,                                \
                                   tile_size_x,                                \
                                   -1])
        return cor_stream_t


    def _temporal_stack_assemble_tile(self,
                                      min_height_OPT, min_width_OPT,
                                      min_height_SAR, min_width_SAR,
                                      num_tiles_x, num_tiles_y,
                                      list_time_stamps,
                                      steps_list,
                                      tfr_tile_files):
        import numpy as np
        import tensorflow as tf
        from datetime import timezone
        from eolearn.core import EOPatch


        def _floats_feature(value):
            return tf.train.Feature(
                                float_list=tf.train.FloatList(value=[value]))


        def _bytes_feature(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[value]))


        prev_frame_OPT = np.zeros((min_height_OPT,
                                  min_width_OPT,
                                  self.bands_opt), dtype=np.float32)
        prev_frame_OPT_t = self._tile_stream(
                prev_frame_OPT,
                min_height_OPT,
                min_width_OPT,
                self.tile_size_y,
                self.tile_size_x,
                self.bands_opt)
        prev_frame_SAR_ascending = np.zeros((min_height_SAR,
                                            min_width_SAR,
                                            self.bands_sar), dtype=np.float32)
        prev_frame_SAR_ascending_t = self._tile_stream(
                prev_frame_SAR_ascending,
                min_height_SAR,
                min_width_SAR,
                self.tile_size_y,
                self.tile_size_x,
                self.bands_sar)
        prev_frame_SAR_descending = np.zeros((min_height_SAR,
                                             min_width_SAR,
                                             self.bands_sar), dtype=np.float32)
        prev_frame_SAR_descending_t = self._tile_stream(
                prev_frame_SAR_descending,
                min_height_SAR,
                min_width_SAR,
                self.tile_size_y,
                self.tile_size_x,
                self.bands_sar)

        idx = 0
        print("List of observations to process:")
        for timestep in list_time_stamps:
            new_sar_asc = False
            new_sar_dsc = False
            new_opt = False
            for item in timestep[1]: # Expected is one item due to no collisions
                if item[0] == self._RS_Type.SAR_ASC:
                    new_patch = EOPatch.load(item[1])
                    new_frame = np.where(                                      \
                        new_patch.mask[self.sar_mask_name]                     \
                                      [0, :min_height_SAR, :min_width_SAR, 0:1],
                        new_patch.data[self.sar_bands_name]                    \
                                      [0,
                                       :min_height_SAR,
                                       :min_width_SAR,
                                       0:self.bands_sar],
                        prev_frame_SAR_ascending[:min_height_SAR,              \
                                                :min_width_SAR,
                                                0:self.bands_sar])
                    prev_frame_SAR_ascending = new_frame
                    new_sar_asc = True
                elif item[0] == self._RS_Type.SAR_DSC:
                    new_patch = EOPatch.load(item[1])
                    new_frame = np.where(
                        new_patch.mask[self.sar_mask_name]                     \
                                      [0, :min_height_SAR, :min_width_SAR, 0:1],
                        new_patch.data[self.sar_bands_name]                    \
                                      [0,
                                       :min_height_SAR,
                                       :min_width_SAR,
                                       0:self.bands_sar],
                        prev_frame_SAR_descending[:min_height_SAR,             \
                                                 :min_width_SAR,
                                                 0:self.bands_sar])
                    prev_frame_SAR_descending = new_frame
                    new_sar_dsc = True
                elif item[0] == self._RS_Type.OPT:
                    new_patch = EOPatch.load(item[1])
                    new_frame = np.where(
                        new_patch.mask[self.opt_mask_name]                     \
                                      [0, :min_height_OPT, :min_width_OPT, 0:1],
                        new_patch.data[self.opt_bands_name]                    \
                                      [0,
                                       :min_height_OPT,
                                       :min_width_OPT,
                                       0:self.bands_opt],
                        prev_frame_OPT[:min_height_OPT,                        \
                                      :min_width_OPT,
                                      0:self.bands_opt])
                    prev_frame_OPT = new_frame
                    new_opt = True
                else:
                    assert False, "Unknown type"
            print(timestep[0])
            # Only write end of time_step
            if timestep[0] not in steps_list:
                continue

            # Try to only compute what's really changed to speed up processing
            if new_sar_asc:
                prev_frame_SAR_ascending_t = self._tile_stream(
                        prev_frame_SAR_ascending,
                        min_height_SAR,
                        min_width_SAR,
                        self.tile_size_y,
                        self.tile_size_x,
                        self.bands_sar)

            if new_sar_dsc:
                prev_frame_SAR_descending_t = self._tile_stream(
                        prev_frame_SAR_descending,
                        min_height_SAR,
                        min_width_SAR,
                        self.tile_size_y,
                        self.tile_size_x,
                        self.bands_sar)

            if new_opt:
                prev_frame_OPT_t = self._tile_stream(
                        prev_frame_OPT,
                        min_height_OPT,
                        min_width_OPT,
                        self.tile_size_y,
                        self.tile_size_x,
                        self.bands_opt)

            # Write all tiles
            tf_ts = timestep[0].replace(tzinfo=timezone.utc).timestamp()
            ts_serial = tf.io.serialize_tensor(tf.convert_to_tensor(
                                                               tf_ts,
                                                               tf.float64))

            for j in range(0, num_tiles_y):
                for i in range(0, num_tiles_x):
                    sar_ascending_serial=tf.io.serialize_tensor(
                            prev_frame_SAR_ascending_t[j, i, :, :, :])
                    sar_descending_serial=tf.io.serialize_tensor(
                            prev_frame_SAR_descending_t[j, i, :, :, :])
                    opt_serial=tf.io.serialize_tensor(
                            prev_frame_OPT_t[j, i, :, :, :])

                    sample_record = {
                        "Timestamp": _bytes_feature(ts_serial),
                        "SAR_ascending": _bytes_feature(sar_ascending_serial),
                        "SAR_descending": _bytes_feature(sar_descending_serial),
                        "OPT": _bytes_feature(opt_serial)
                    }

                    sample = tf.train.Example(
                            features=tf.train.Features(feature=sample_record))
                    tfr_tile_files[j][i].write(sample.SerializeToString())

            idx += 1
        print("Total writes: ", idx)


    def process(self):
        """
        Start the stacking, assembling and tiling process.

        :rtype: None

        **Example:**

        Start the stacking, assembling, and tiling process as defined by
        the object ``stack``. This will require significant compute resources
        depending on the amount of stacked observations.

        .. code-block:: python

            stack.process()

        """
        import tensorflow as tf
        import json

        all_files_OPT = self._getEOPatches(self.opt_path)
        all_files_SAR_ascending = self._getEOPatches(self.sar_asc_path)
        all_files_SAR_descending = self._getEOPatches(self.sar_dsc_path)

        min_height_OPT, min_width_OPT,                                         \
        min_height_SAR, min_width_SAR = self._get_min_resolution(
                                            all_files_OPT,
                                            all_files_SAR_ascending,
                                            all_files_SAR_descending)

        print("Resolutions (y, x):")
        print("\tOPT: {}, {}".format(min_height_OPT, min_width_OPT))
        print("\tSAR: {}, {}".format(min_height_SAR, min_width_SAR))

        list_time_stamps = []
        for typ in all_files_OPT +                                             \
                   all_files_SAR_ascending +                                   \
                   all_files_SAR_descending:
            this_type = self._RS_Type.OPT if typ in all_files_OPT else         \
                        self._RS_Type.SAR_ASC if typ in                        \
                                                all_files_SAR_ascending else   \
                        self._RS_Type.SAR_DSC if typ in                        \
                                                all_files_SAR_descending else  \
                        self._RS_Type.UNKOWN

            match_idx = [idx for idx, x in enumerate(list_time_stamps)         \
                         if x[0] == typ[0]]

            if match_idx:
                # Technically there should not be two observations with the
                # same time stamp.
                assert len(match_idx) < 2, "Too many matches (>1)!"
                list_time_stamps[match_idx[0]][1].append((this_type, typ[1]))
            else:
                list_time_stamps.append((typ[0], [(this_type, typ[1])]))

        list_time_stamps.sort(key=lambda tup: tup[0])

        print("Total time stamps: {}".format(len(list_time_stamps)))

        steps_list = self._get_steps_list(list_time_stamps)

        num_tiles_y = (min_height_SAR - self.deadzone_y)//                     \
                                        (self.tile_size_y - 2*self.deadzone_y)
        num_tiles_x = (min_width_SAR - self.deadzone_x)//                      \
                                        (self.tile_size_x - 2*self.deadzone_x)

        # Open all TFRecord files for writing
        tfr_options = tf.io.TFRecordOptions(compression_type="GZIP")
        tfr_tile_files = []
        for j in range(0, num_tiles_y):
            tfr_tile_files.append([])
            for i in range(0, num_tiles_x):
                tfr_tile_files[j].append(
                        tf.io.TFRecordWriter(self.tf_record_path +             \
                            "{}_{}.tfrecords".format(j, i),
                        options=tfr_options))


        self._temporal_stack_assemble_tile(min_height_OPT, min_width_OPT,
                             min_height_SAR, min_width_SAR,
                             num_tiles_x, num_tiles_y,
                             list_time_stamps,
                             steps_list,
                             tfr_tile_files)

        # Close all TFRecords
        for j in range(0, num_tiles_y):
            for i in range(0, num_tiles_x):
                tfr_tile_files[j][i].close()

        # Write metadata
        print("Writing metadata")
        metadata_dict = {
            "tile_size_y": self.tile_size_y,
            "tile_size_x": self.tile_size_x,
            "bands_sar": self.bands_sar,
            "bands_opt": self.bands_opt,
            "deadzone_y": self.deadzone_y,
            "deadzone_x": self.deadzone_x
        }
        json_object = json.dumps(metadata_dict, indent=2)
        with open(self.tf_record_path +                                        \
                  "metadata.json", "w") as metadata_json_file:
            metadata_json_file.write(json_object)


class Window:
    """
    Class for windowing the time series of observations.

    :param tf_record_path: Path to stacked TFRecord files
    :type tf_record_path: str
    :param delta_size: Window size in seconds (:math:`\\Delta`)
    :type delta_size: int
    :param window_stride: Stride of window (:math:`\\rho`)
    :type window_stride: int
    :param omega: Minimum window size in number of observations (:math:`\\omega`)
    :type omega: int
    :param Omega: Maximum window size in number of observations (:math:`\\Omega`)
    :type Omega: int
    :param generate_triple: Indicate whether a window triplet should be
        generated or just a single window
    :type generate_triple: boolean
    :param n_threads: Number of threads to use for concurrent processing
        (default = ``1``)
    :type n_threads: int
    :param use_new_save: Whether to use ``tf.data.Dataset.save(...)`` from
        Tensorflow if ``true``. If ``flase`` (default) create TFRecord files.
    :type use_new_save: boolean

    **Example:**

    Define the windowing (and optional labeling) with different parameters. This
    can be run in two modes:

        - Write to disk: `write_tf_files(...)`
        - Interactive: `get_infer_dataset(...)`

    Irrespective of the modes, the windows are described with the class
    instantiation. In the example below, the path to the stacked TFRecord files
    are provided as ``tf_record_path``. The window parameters :math:`\Delta` for
    the window size in seconds, the window stride :math:`\\rho`, and lower
    (:math:`\\omega`) and upper bound (:math:`\\Omega`) of number of
    observations per window are provided here.

    .. code-block:: python

        import rsdtlib

        tf_record_path = "<SOURCE PATH STACKED TFRECORD FILES>"

        window = rsdtlib.Window(
                      tf_record_path,        # stacked TFRecord file path
                      60*60*24*30,           # Delta (size)
                      1,                     # window stride
                      10,                    # omega (min. window size)
                      16,                    # Omega (max. window size)
                      True,                  # generate triplet
                      n_threads=n_threads,   # number of threads to use
                      use_new_save=False)    # new TF Dataset save

    """
    def __init__(self,
                 tf_record_path,
                 delta_size,
                 window_stride,
                 omega,
                 Omega,
                 generate_triple,
                 n_threads = 1,
                 use_new_save = False):
        import math
        import json

        self.tf_record_path = tf_record_path
        self.delta_size = delta_size
        self.window_stride = window_stride
        self.omega = omega
        self.Omega = Omega
        self.generate_triple = generate_triple
        self.n_threads = n_threads
        self.use_new_save = use_new_save

        with open(self.tf_record_path +                                        \
                  "metadata.json", "r") as metadata_json_file:
            json_object = json.load(metadata_json_file)
            self.tile_size_y = json_object["tile_size_y"]
            self.tile_size_x = json_object["tile_size_x"]
            self.bands_sar = json_object["bands_sar"]
            self.bands_opt = json_object["bands_opt"]

#        if self.use_new_save == False:
#            assert self.n_threads == 1, "TFRecordWriter only allows one thread!"


    def _parse_tfr_element(self, element):
        import tensorflow as tf

        parse_dic = {
            "Timestamp": tf.io.FixedLenFeature([], tf.string),
            "SAR_ascending": tf.io.FixedLenFeature([], tf.string),
            "SAR_descending": tf.io.FixedLenFeature([], tf.string),
            "OPT": tf.io.FixedLenFeature([], tf.string)
        }
        example_message = tf.io.parse_single_example(element, parse_dic)

        timestamp_f = example_message["Timestamp"]
        timestamp = tf.ensure_shape(tf.io.parse_tensor(timestamp_f,
                                                       out_type=tf.float64),
                                    None)

        sar_ascending_f = example_message["SAR_ascending"]
        sar_ascending = tf.ensure_shape(tf.io.parse_tensor(sar_ascending_f,
                                                          out_type=tf.float32),
                                        [self.tile_size_y,
                                         self.tile_size_x,
                                         self.bands_sar])

        sar_descending_f = example_message["SAR_descending"]
        sar_descending = tf.ensure_shape(tf.io.parse_tensor(sar_descending_f,
                                                           out_type=tf.float32),
                                         [self.tile_size_y,
                                          self.tile_size_x,
                                          self.bands_sar])

        opt_f = example_message["OPT"]
        opt = tf.ensure_shape(tf.io.parse_tensor(opt_f, out_type=tf.float32),
                              [self.tile_size_y,
                               self.tile_size_x,
                               self.bands_opt])

        return timestamp, sar_ascending, sar_descending, opt


    def _get_window(self, timestamp, sar_ascending, sar_descending, opt):
        import tensorflow as tf

        sub = tf.data.Dataset.zip((timestamp,
                                   sar_ascending,
                                   sar_descending,
                                   opt))
        batch = sub.batch(self.Omega)
        return batch


    def _get_window2(self, timestamp, sar_ascending, sar_descending, opt):
        import tensorflow as tf

        sub = tf.data.Dataset.zip((timestamp,
                                   sar_ascending,
                                   sar_descending,
                                   opt))
        batch = sub.batch(self.Omega*2)
        return batch


    def _trunc_windows_for_triple(self, tpls):
        import tensorflow as tf

        @tf.function
        def broadcast_first_timestep(timesteps_prev, timesteps_cur):
            return tf.broadcast_to(timesteps_cur[0], [tf.size(timesteps_prev)])

        @tf.function
        def get_subset_prev(x, y):
            this_mask = tf.math.logical_and(
                            tf.math.greater_equal(x[0], y-self.delta_size),
                            tf.math.less(x[0], y))
            this_mask.set_shape(y.get_shape())
            return ((tf.boolean_mask(x[0], this_mask),
                     tf.boolean_mask(x[1], this_mask),
                     tf.boolean_mask(x[2], this_mask),
                     tf.boolean_mask(x[3], this_mask)))

        @tf.function
        def get_subset_cur(x, y):
            this_mask = tf.math.logical_and(
                            tf.math.greater_equal(x[0], y),
                            tf.math.less(x[0], y+self.delta_size))
            this_mask.set_shape(y.get_shape())
            return ((tf.boolean_mask(x[0], this_mask),
                     tf.boolean_mask(x[1], this_mask),
                     tf.boolean_mask(x[2], this_mask),
                     tf.boolean_mask(x[3], this_mask)))

        @tf.function
        def get_subset_next(x, y):
            this_mask = tf.math.logical_and(
                            tf.math.greater_equal(x[0], y+self.delta_size),
                            tf.math.less(x[0], y+2*self.delta_size))
            this_mask.set_shape(y.get_shape())
            return ((tf.boolean_mask(x[0], this_mask),
                     tf.boolean_mask(x[1], this_mask),
                     tf.boolean_mask(x[2], this_mask),
                     tf.boolean_mask(x[3], this_mask)))

        prev = tpls.map(lambda x, y: x)
        cur_next = tpls.map(lambda x, y: y)

        # Select samples from previous window
        timesteps_prev = prev.map(lambda a, b, c, d: a)
        timesteps_cur_next = cur_next.map(lambda a, b, c, d: a)
        ref_prev = tf.data.Dataset.zip((timesteps_prev, timesteps_cur_next))   \
                                  .map(broadcast_first_timestep)

        selection_prev = tf.data.Dataset.zip((prev, ref_prev))
        prev_win = selection_prev.map(get_subset_prev)

        # Select samples from current and following window
        ref_cur = tf.data.Dataset.zip((timesteps_cur_next, timesteps_cur_next))\
                                 .map(broadcast_first_timestep)

        selection_cur_next = tf.data.Dataset.zip((cur_next, ref_cur))
        cur_win = selection_cur_next.map(get_subset_cur)
        next_win = selection_cur_next.map(get_subset_next)

        return tf.data.Dataset.zip((prev_win, cur_win, next_win))


    def _trunc_windows_for_mono(self, cur):
        import tensorflow as tf

        @tf.function
        def broadcast_first_timestep(timesteps_prev, timesteps_cur):
            return tf.broadcast_to(timesteps_cur[0], [tf.size(timesteps_prev)])

        @tf.function
        def get_subset_cur(x, y):
            this_mask = tf.math.logical_and(
                            tf.math.greater_equal(x[0], y),
                            tf.math.less(x[0], y+self.delta_size))
            this_mask.set_shape(y.get_shape())
            return ((tf.boolean_mask(x[0], this_mask),
                     tf.boolean_mask(x[1], this_mask),
                     tf.boolean_mask(x[2], this_mask),
                     tf.boolean_mask(x[3], this_mask)))

        # Select samples from current window
        timesteps_cur = cur.map(lambda a, b, c, d: a)
        ref_cur = tf.data.Dataset.zip((timesteps_cur, timesteps_cur))          \
                                 .map(broadcast_first_timestep)

        selection_cur = tf.data.Dataset.zip((cur, ref_cur))
        cur_win = selection_cur.map(get_subset_cur)

        return tf.data.Dataset.zip((cur_win, ))

    def _annotate_ds(self, sample_file):
        import tensorflow as tf

        # Workaraound to get rid of warning reg. AUTOGRAPH
        filter_obs_for_triple = tf.autograph.experimental.do_not_convert(
                            lambda prev_win, cur_win, next_win:                \
                            tf.math.logical_and(                               \
                                tf.math.logical_and(                           \
                                    tf.shape(prev_win[0])[0] >= self.omega,    \
                                    tf.shape(cur_win[0])[0] >= self.omega),    \
                                    tf.shape(next_win[0])[0] >= self.omega))

        filter_obs_for_mono = tf.autograph.experimental.do_not_convert(
                            lambda cur_win:                                    \
                            tf.shape(cur_win[0])[0] >= self.omega)

        input_ds = tf.data.TFRecordDataset(sample_file,
                                         compression_type="GZIP",
                                         num_parallel_reads=1)                 \
                          .map(self._parse_tfr_element, num_parallel_calls=1)

        if self.generate_triple:
            # Construct previous window (starting from [] up to [Omega]
            dataset_enum = tf.data.Dataset.from_tensor_slices(
                                                tf.range(
                                                    0,
                                                    self.Omega,
                                                    dtype=tf.int64))
            tmp_dataset = tf.data.Dataset.zip(
                (dataset_enum,
                 input_ds.take(self.Omega)                                     \
                    .batch(self.Omega).repeat(self.Omega)))
            tmp_dataset = tmp_dataset.map(lambda x, y: (y[0][0:x],  # Timestamp
                                                        y[1][0:x],  # SAR asc.
                                                        y[2][0:x],  # SAR dsc.
                                                        y[3][0:x])) # OPT

            prev_window_ds = tmp_dataset.concatenate(
                input_ds.window(
                            self.Omega,
                            shift=self.window_stride,
                            stride=1)                                          \
                        .flat_map(self._get_window))

            # Construct current and next windows
            cur_next_window_ds = input_ds.window(
                                            self.Omega*2,
                                            shift=self.window_stride,
                                            stride=1)                          \
                                         .flat_map(self._get_window2)

            # Concatenate both together and filter for time frames of delta_size
            comb_dataset = tf.data.Dataset.zip((prev_window_ds,
                                                cur_next_window_ds))
            comb_dataset = comb_dataset.apply(self._trunc_windows_for_triple)

            # Return only windows that have at least omega observations
            res_dataset = comb_dataset.filter(filter_obs_for_triple)

        else: # self.generate_triple == False
            cur_window_ds = input_ds.window(
                                    self.Omega,
                                    shift=self.window_stride,
                                    stride=1)                                  \
                    .flat_map(self._get_window)
            comb_dataset = cur_window_ds.apply(self._trunc_windows_for_mono)

            # Return only windows that have at least omega observations
            res_dataset = comb_dataset.filter(filter_obs_for_mono)

        return res_dataset


    def windows_list(self):
        """
        Retrieve the list of windows without constructing them.

        :return: Returns a list of window descriptors
        :rtype: Each window is described as a quadruple
            ``(id, starttime, enddtime, no_obs)``:

                - ``id``: ID of the window (zero based, sequential enumeration)
                - ``starttime``: Time stamp of first observation in window
                - ``endtime``: Time stamp of last observation in window
                - ``no_obs``: Number of observations in window

        **Example:**

        In the example below, the list of windows that are generated are saved
        to a CSV file.

        .. code-block:: python

            import rsdtlib
            import csv

            window_list = window.windows_list()

            with open("windows_training.csv", mode = "w") as csv_file:
                csv_writer = csv.writer(csv_file,
                                        delimiter=",",
                                        quotechar="\\"",
                                        quoting=csv.QUOTE_MINIMAL)
                for item in window_list:
                    csv_writer.writerow([item[0],
                                         datetime.utcfromtimestamp(item[1]),
                                         datetime.utcfromtimestamp(item[2]),
                                         item[3]])

        """
        import datetime
        import tensorflow as tf

        # Just one tile as representative...
        j = 0
        i = 0
        sample_file = self.tf_record_path +                                    \
                      "{}_{}.tfrecords".format(j, i)
        get_windows_ds = self._annotate_ds(sample_file)

        if self.generate_triple:
            print("List of window ranges (previous, current, next):")
            for item in get_windows_ds:
                if tf.shape(item[0][0]) > 0:
                    window_start = datetime.datetime.utcfromtimestamp(
                                                    item[0][0][0].numpy())
                    window_end = datetime.datetime.utcfromtimestamp(
                                                    item[0][0][-1].numpy())

                    print(window_start, window_end)
                else:
                    print("EMPTY")
                if tf.shape(item[1][0]) > 0:
                    window_start = datetime.datetime.utcfromtimestamp(
                                                    item[1][0][0].numpy())
                    window_end = datetime.datetime.utcfromtimestamp(
                                                    item[1][0][-1].numpy())

                    print("\t", window_start, window_end)
                else:
                    print("\tEMPTY")
                if tf.shape(item[2][0]) > 0:
                    window_start = datetime.datetime.utcfromtimestamp(
                                                    item[2][0][0].numpy())
                    window_end = datetime.datetime.utcfromtimestamp(
                                                    item[2][0][-1].numpy())

                    print("\t\t", window_start, window_end)
                else:
                    print("\t\tEMPTY")
                print()

        else: # self.generate_triple == False
            print("List of window ranges (current):")
            for item in get_windows_ds:
                if tf.shape(item[0][0]) > 0:
                    window_start = datetime.datetime.utcfromtimestamp(
                                                    item[0][0][0].numpy())
                    window_end = datetime.datetime.utcfromtimestamp(
                                                    item[0][0][-1].numpy())

                    print(window_start, window_end)
                else:
                    print("EMPTY")

        this_id = 0
        windows_list = []
        for item in get_windows_ds:
            if self.generate_triple:
                this_item = item[1]
            else: # self.generate_triple == False
                this_item = item[0]

            window_amount = len(this_item[0].numpy())
            windows_list.append((this_id,
                                int(this_item[0][0].numpy()),
                                int(this_item[0][-1].numpy()),
                                window_amount))
            this_id += 1
        return windows_list


    def get_num_tiles(self):
        """
        Get the number of y-x tiles. It assumes that no gap of tiles exist.

        Note: Omitting tiles is possible. This function only takes the maximum
        y-x tile coordinates. In further processing a selector can be used to
        filter non-available tiles.

        :return: Returns a tuple ``(y, x)``
        :rtype: Tuple of ``(int, int)``

        **Example:**

        In the example below, the amount of tiles in each dimension are
        returned.

        .. code-block:: python

            import rsdtlib

            num_tiles_y, num_tiles_x = window.get_num_tiles()

       """
        import os
        import re

        tiles = [f for f in os.listdir(self.tf_record_path)
                     if os.path.isfile(os.path.join(self.tf_record_path, f)) and
                        f.endswith(".tfrecords")]
        num_tiles_y = -1
        num_tiles_x = -1
        tile_pattern = re.compile("^([0-9]+)_([0-9]+).tfrecords$")
        for tle in tiles:
            match = tile_pattern.match(tle)
            assert match, "TFRecord file does not have expected pattern: " + tle
            this_y = int(match[1])
            this_x = int(match[2])
            num_tiles_y = this_y if this_y > num_tiles_y else num_tiles_y
            num_tiles_x = this_x if this_x > num_tiles_x else num_tiles_x

        # Tiles are zero-based
        num_tiles_y += 1
        num_tiles_x += 1
        return num_tiles_y, num_tiles_x


    def _write_training_data(self,
                             tf_record_out_file,
                             dataset,
                             label_args_ds,
                             gen_label):
        import tensorflow as tf

        def _bytes_feature(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value]))


        comb_ds = tf.data.Dataset.zip((dataset, label_args_ds))
        comb_ds = comb_ds.map(gen_label)
        if self.use_new_save:
            tf.data.Dataset.save(comb_ds, tf_record_out_file, "GZIP")
#            tf.data.experimental.save(comb_ds, tf_record_out_file, "GZIP")

        else: # self.use_new_save == False
            tfr_options = tf.io.TFRecordOptions(compression_type="GZIP")
            this_tfr = tf.io.TFRecordWriter(
                                tf_record_out_file,
                                options=tfr_options)

            for item in comb_ds:
                feature = tf.io.serialize_tensor(item[0])
                label = tf.io.serialize_tensor(item[1])

                sample_record = {
                    "Feature": _bytes_feature(feature),
                    "Label": _bytes_feature(label)
                }

                sample = tf.train.Example(features=tf.train.Features(
                                                        feature=sample_record))
                this_tfr.write(sample.SerializeToString())

            this_tfr.close()


    def _write_inference_data(self, tf_record_out_file, dataset):
        import tensorflow as tf

        def _get_sample(data):
            res = tf.concat(
                        # Only serialize current window (index 1)
                        [data[1][:, :, :, :],  # SAR ascending
                         data[2][:, :, :, :],  # SAR descending
                         data[3][:, :, :, :]], # Optical
                         axis=-1)
            return res


        def _bytes_feature(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value]))


        comb_ds = dataset.map(_get_sample)
        if self.use_new_save:
            tf.data.Dataset.save(comb_ds, tf_record_out_file, "GZIP")
#            tf.data.experimental.save(comb_ds, tf_record_out_file, "GZIP")

        else: # self.use_new_save == False
            tfr_options = tf.io.TFRecordOptions(compression_type="GZIP")
            this_tfr = tf.io.TFRecordWriter(
                                tf_record_out_file,
                                options=tfr_options)

            for item in comb_ds:
                feature = tf.io.serialize_tensor(item[0])

                sample_record = {
                    "Feature": _bytes_feature(feature),
                }

                sample = tf.train.Example(features=tf.train.Features(
                                                        feature=sample_record))
                this_tfr.write(sample.SerializeToString())

            this_tfr.close()


    def write_tf_files(self,
                       tf_record_out_path,
                       selector,
                       win_filter=None,
                       label_args_ds=None,
                       gen_label=None):
        """
        Write the windows as TFRecord files (one for each tile).

        :param tf_record_out_path: Path to destination where to store the
            TFRecord files
        :type tf_record_out_path: str
        :param selector: Functor to query which tile should be written
        :type slector: selector(y, x) -> boolean
        :param win_filter: Filter for windows (default = ``None``)
        :type win_filter: tf.data.Dataset.filter predicate
        :param label_args_ds: Arguments for labeling (default = ``None``). The
            sequence has to be identical to the samples.
        :type label_args_ds: tf.data.Dataset
        :param gen_label: Functor to generate the labels (default = ``None``).
            Training data is provided by ``data``, the label arguments via
            ``label_args``. The output is a label in spatial ``y`` and ``x``.
            dimension.
        :type gen_label: gen_label(data, label_args) -> [y, x]
        :return: None

        **Example:**

        This is the windowing mode to write to disk. In the example below,
        the path to the windowed TFRecord files are provided as
        ``tf_record_out_path``. A checkerboard pattern of tiles are windowed.
        This is useful for separating training and validation/testing data.
        The ``win_filter`` allows to add additional filters for windows. In the
        example, randomly every 10th window is saved and the rest is discarded.
        If labels should be assigned to every window, ``gen_label`` is the
        generator for these labels. In the example, only labels with values of
        one are created.

        .. code-block:: python

            import rsdtlib
            import tensorflow as tf

            tf_record_out_path = "<DESTINAITON PATH OF WINDOWED TFRECORD FILES>"

            selector = lambda j, i: (i + j) % 2 == 0 # checkerboard pattern

            # Randomly select every 10th window only
            randomize = lambda *args:                                         \\
                            tf.random.uniform([], 0, 10,
                                              dtype=tf.dtypes.int32) == 0

            # Generate 32x32 pixel label with only values of one
            my_label = lambda window, label_args:                             \\
                  (tf.concat(
                        # Only serialize current window (index 1)
                        # Note: We require window triplets are generated!
                        [
                         #window[1][0][:],          # timestamps (not used)
                         window[1][1][:, :, :, :],  # SAR ascending
                         window[1][2][:, :, :, :],  # SAR descending
                         window[1][3][:, :, :, :]], # optical
                        axis=-1),
                   tf.ensure_shape(                 # label
                        tf.ones([32, 32]), [32, 32]))

            window.write_tf_files(
                          tf_record_out_path, # path to write TFRecord files to
                          selector,
                          win_filter=randomize,
                          gen_label=my_label)

       """
        import os
        import datetime
        import tensorflow as tf

        def write_data(tile, win_filter, label_args_ds, gen_label):
            sample_file = self.tf_record_path +                                \
                          "{}_{}.tfrecords".format(tile[0], tile[1])
            windows_ds = self._annotate_ds(sample_file)

            if win_filter is not None:
                windows_ds = windows_ds.filter(win_filter)

            if label_args_ds is not None:
                self._write_training_data(
                                tf_record_out_path +                           \
                                "{}_{}.tfrecords".format(tile[0], tile[1]),    \
                                windows_ds,                                    \
                                label_args_ds,                                 \
                                gen_label)

            else: # label_args_ds is None
                self._write_inference_data(
                                tf_record_out_path +                           \
                                "{}_{}.tfrecords".format(tile[0], tile[1]),    \
                                windows_ds)
            return 0


        # Note: If multi-threaded by caller, this test might not work properly.
        #       Hence, we ignore cases where the directory already exists!
        if not os.path.isdir(tf_record_out_path):
            try:
                os.mkdir(tf_record_out_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        num_tiles_y, num_tiles_x = self.get_num_tiles()

        list_tiles = []
        for j in range(0, num_tiles_y):
            for i in range(0, num_tiles_x):
                if selector(j, i):
                    list_tiles.append((j, i))

# FIXME: How to pass all new arguments?
#        write_ds = tf.data.Dataset.from_generator(lambda: list_tiles,
#                                                  output_signature=(
#                                                      tf.TensorSpec(
#                                                          shape=(2),
#                                                          dtype=tf.uint64)))
#        # Note: Multi-threading is done externally since writing TFRecord
#        #       files is not well parallelizable.
#        write_ds = write_ds.map(lambda tile: tf.py_function(
#                                        write_data,
#                                        [tile],
#                                        [tf.uint64]),
#                                        num_parallel_calls=1)
#
#        for item in write_ds:
#            pass # Just iterate to write the files
        for item in list_tiles:
            write_data(item, win_filter, label_args_ds, gen_label)

        return


    def get_infer_dataset(self, tile, win_filter=None):
        """
        Return a dataset for inference.

        :param tile: Tile coordinates in ``y`` and ``x`` dimensions
        :type tile: [y, x]
        :param win_filter: Filter for windows (default = ``None``)
        :type win_filter: tf.data.Dataset.filter predicate
        :return: If tile exists, return the dataset, otherwise ``None``
        :rtype: ``tf.data.Dataset`` | ``None``

        **Example:**

        This is the interactive mode. In the example below, the windows of the
        tile ``[5, 10]`` are used for inference. The ``win_filter`` allows to
        add additional filters for windows. In the example, only windows that
        have their first observation later than 2022-07-01 00:00:00 (GMT) are
        considered. Note that ``window[0][0]`` denotes the first timestamp.
        When applying filters, there is no notion of previous, current or next
        windows (irrespective of the setting of ``generate_triple``).

        .. code-block:: python

            import rsdtlib
            import tensorflow as tf

            later_than = lambda window: tf.math.greater(
                                            tf.cast(window[0][0], tf.int64),
                                            1656626400) # 2022-07-01 00:00:00

            windows_ds = window.get_infer_dataset([5, 10],
                                                  win_filter=later_than)

            # Use for inference on a loaded Tensorflow/Keras model
            result = model.predict(windows_ds)

       """
        import os
        import datetime
        import tensorflow as tf

        sample_file = self.tf_record_path +                                    \
                      "{}_{}.tfrecords".format(tile[0], tile[1])
        if os.path.exists(sample_file):
            windows_ds = self._annotate_ds(sample_file)

            if win_filter is not None:
                windows_ds = windows_ds.filter(win_filter)

            return windows_ds

        return None
