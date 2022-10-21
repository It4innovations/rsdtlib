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

class Retrieve:
    def __init__(self, starttime, endtime, aoi, shconfig):
        self.starttime = starttime
        self.endtime = endtime
        self.aoi = aoi
        self.shconfig = shconfig


    def _get_intervall(self):
        """
            Returns monthly intervals to the provided time range
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
            Return the bounding box defined by the shapefile (i.e. AoI)

            Note: Only the first polygon in the shapefile will be used!
        """
        import geopandas as gpd
        from sentinelhub import geometry

        shp_data = gpd.read_file(self.aoi)
        shp_data.crs = "EPSG:4326"

        WGS84_data = shp_data.to_crs("EPSG:4326")
        for index, row in WGS84_data.iterrows():
            return geometry.Geometry(row["geometry"], crs="EPSG:4326").bbox


    def _merge_CLM(self,
                   eopatches_tmp_dir,
                   eopatches_clm_dir,
                   eopatches_out_dir,
                   eopatches_fail_dir):
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
        import os
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
            print("Error: Temporary directory already exists: {}".format(
                                                             eopatches_tmp_dir))
            return
        else:
            os.mkdir(eopatches_tmp_dir)

        if datacollection == DataCollection.SENTINEL2_L1C:
            eopatches_clm_dir = "{}/CLM_eopatches/".format(dst_path)
            if os.path.isdir(eopatches_clm_dir):
                print("Error: Temporary directory already exists: {}".format(
                                                             eopatches_clm_dir))
                return
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

        # For optical data, download also cloud masks and merge. If that fails,
        # move to failed directory.
        # For SAR data, test if observations are empty and move those to failed
        # directory.
        if datacollection == DataCollection.SENTINEL2_L1C:
            get_clm_task = SentinelHubInputTask(
                data_collection = datacollection,
                bands_feature = None,
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

            for i in all_samples:
                start_d = i - datetime.timedelta(0,1)
                end_d = i + datetime.timedelta(0,1)
                this_time_str = i.strftime("%Y%m%dT%H%M%S")
                if (os.path.isdir("{}/{}".format(eopatches_out_dir,
                                                this_time_str)) or
                    os.path.isdir("{}/{}".format(eopatches_fail_dir,
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


from enum import Enum
class RS_Type(Enum):
    UNKOWN = -1
    OPT = 0
    SAR_ASC = 1
    SAR_DSC = 2

class Stack:
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
                 bands_opt,
                 bands_sar):
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
        self.bands_opt = bands_opt
        self.bands_sar = bands_sar


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
                           strides=[1, tile_size_y, tile_size_x, 1],           \
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
                if item[0] == RS_Type.SAR_ASC:
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
                elif item[0] == RS_Type.SAR_DSC:
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
                elif item[0] == RS_Type.OPT:
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
        import tensorflow as tf

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
            this_type = RS_Type.OPT if typ in all_files_OPT else               \
                        RS_Type.SAR_ASC if typ in all_files_SAR_ascending else \
                        RS_Type.SAR_DSC if typ in all_files_SAR_descending else\
                        RS_Type.UNKOWN

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

        num_tiles_y = min_height_SAR//self.tile_size_y
        num_tiles_x = min_width_SAR//self.tile_size_x

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


class Window:
    def __init__(self,
                 tf_record_path,
                 tf_record_out_path,
                 delta_size,
                 window_shift,
                 omega,
                 Omega,
                 tile_size_x,
                 tile_size_y,
                 bands_opt,
                 bands_sar,
                 generate_labels,
                 alpha = None,
                 n_threads = 1,
                 use_new_save = False):
        import math

        self.tf_record_path = tf_record_path
        self.tf_record_out_path = tf_record_out_path
        self.delta_size = delta_size
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.window_shift = window_shift
        self.omega = omega
        self.Omega = Omega
        self.bands_opt = bands_opt
        self.bands_sar = bands_sar
        self.generate_labels = generate_labels
        self.alpha = alpha
        self.n_threads = n_threads
        self.use_new_save = use_new_save

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


    def _get_windows(self, timestamp, sar_ascending, sar_descending, opt):
        import tensorflow as tf

        def first_only(batch):
            return tf.broadcast_to(batch[0], [tf.size(batch)])

        ref = timestamp.batch(self.Omega).map(first_only)

        sub = tf.data.Dataset.zip((timestamp,
                                   sar_ascending,
                                   sar_descending,
                                   opt))
        batch = sub.batch(self.Omega)
        selection = tf.data.Dataset.zip((batch, ref))
        windows = selection.unbatch()                                          \
                           .filter(lambda x, y:                                \
                                   tf.math.less(x[0], y + self.delta_size))    \
                           .map(lambda x, y: x)                                \
                           .batch(self.Omega)

        return windows


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


    def _trunc_windows_for_labels(self, tpls):
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


    def _trunc_windows_no_labels(self, cur):
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

    def _annotate_ds(self, sample_file, window_betas = None):
        import tensorflow as tf

        # Workaraound to get rid of warning reg. AUTOGRAPH
        filter_obs_for_labels = tf.autograph.experimental.do_not_convert(
                            lambda prev_win, cur_win, next_win:                \
                            tf.math.logical_and(                               \
                                tf.math.logical_and(                           \
                                    tf.shape(prev_win[0])[0] >= self.omega,    \
                                    tf.shape(cur_win[0])[0] >= self.omega),    \
                                    tf.shape(next_win[0])[0] >= self.omega))

        filter_obs_no_labels = tf.autograph.experimental.do_not_convert(
                            lambda cur_win:                                    \
                            tf.shape(cur_win[0])[0] >= self.omega)

        # Workaraound to get rid of warning reg. AUTOGRAPH
        filter_rand = tf.autograph.experimental.do_not_convert(
                            lambda *args:                                      \
                            tf.random.uniform([],
                                              0,
                                              10,
                                              dtype=tf.dtypes.int32,
                                              seed=seed) == 0)

        input_ds = tf.data.TFRecordDataset(sample_file,
                                         compression_type="GZIP",
                                         num_parallel_reads=1)                 \
                        .map(self._parse_tfr_element, num_parallel_calls=1)

        if self.generate_labels:
            # Construct previous window (starting from [] up to [Omega]
            dataset_enum = tf.data.Dataset.from_tensor_slices(
                                                tf.range(
                                                    0,
                                                    self.Omega,
                                                    dtype=tf.int64))
            tmp_dataset = tf.data.Dataset.zip(
                (dataset_enum,
                 input_ds.take(self.Omega)                               \
                    .batch(self.Omega).repeat(self.Omega)))
            tmp_dataset = tmp_dataset.map(lambda x, y: (y[0][0:x],  # Timestamp
                                                        y[1][0:x],  # SAR asc.
                                                        y[2][0:x],  # SAR dsc.
                                                        y[3][0:x])) # OPT

            prev_window_ds = tmp_dataset.concatenate(
                input_ds.window(
                            self.Omega,
                            shift=self.window_shift,
                            stride=1)                                          \
                    .flat_map(self._get_window))

            # Construct current and next windows
            cur_next_window_ds = input_ds.window(
                                    self.Omega*2,
                                    shift=self.window_shift,
                                    stride=1)                                  \
                    .flat_map(self._get_window2)

            # Concatenate both together and filter for timeframes of delta_size
            comb_dataset = tf.data.Dataset.zip((prev_window_ds,
                                                cur_next_window_ds))
            comb_dataset = comb_dataset.apply(self._trunc_windows_for_labels)

            # Return only windows that have at least omega observations
            res_dataset = comb_dataset.filter(filter_obs_for_labels)

        else: # self.generate_labels == False
            cur_window_ds = input_ds.window(
                                    self.Omega,
                                    shift=self.window_shift,
                                    stride=1)                                  \
                    .flat_map(self._get_window2)
            comb_dataset = cur_window_ds.apply(self._trunc_windows_no_labels)

            # Return only windows that have at least omega observations
            res_dataset = comb_dataset.filter(filter_obs_no_labels)

        return res_dataset


    def windows_list(self):
        import datetime
        import tensorflow as tf

        # Just one tile as representative...
        j = 0
        i = 0
        sample_file = self.tf_record_path +                                    \
                      "{}_{}.tfrecords".format(j, i)
        get_windows_ds = self._annotate_ds(sample_file)

        if self.generate_labels:
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

        else: # self.generate_labels == False
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
            if self.generate_labels:
                this_item = item[1]
            else: # self.generate_labels == False
                this_item = item[0]

            window_amount = len(this_item[0].numpy())
            windows_list.append((this_id,
                                int(this_item[0][0].numpy()),
                                int(this_item[0][-1].numpy()),
                                window_amount))
            this_id += 1
        return windows_list


    def get_num_tiles(self):
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


    def _write_training_data(self, tf_record_out_file, dataset, window_betas):
        import tensorflow as tf
        import sys
        sys.path.append('../label/')
        from label import Synthetic_Label

        def _get_sample_label(data, betas):
            res = (tf.concat(
                        # Only serialize current window (index 1)
                        [data[1][1][:, :, :, :],  # SAR ascending
                         data[1][2][:, :, :, :],  # SAR descending
                         data[1][3][:, :, :, :]], # Optical
                         axis=-1),
                   tf.ensure_shape(tf.numpy_function(
                        Synthetic_Label.compute_label_S2_S1_ENDISI,
                        [data[1][1][:, :, :, :], # SAR ascending
                         data[1][2][:, :, :, :], # SAR descending
                         data[1][3][:, :, :, :], # Optical
                         data[0][3][:, :, :, :], # Optical (prev)
                         data[2][3][:, :, :, :], # Optical (next)
                         self.alpha,
                         betas[0],
                         betas[1]], tf.float32),
                        [self.tile_size_y, self.tile_size_x]))
            return res

        def _bytes_feature(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[value]))

        betas_ds = tf.data.Dataset.from_tensor_slices(window_betas)
        comb_ds = tf.data.Dataset.zip((dataset, betas_ds))
        comb_ds = comb_ds.map(_get_sample_label)
        if self.use_new_save:
#            tf.data.Dataset.save(comb_ds, tf_record_out_file, "GZIP")
            tf.data.experimental.save(comb_ds, tf_record_out_file, "GZIP")

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
        import sys
        sys.path.append('../label/')
        from label import Synthetic_Label

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
#            tf.data.Dataset.save(comb_ds, tf_record_out_file, "GZIP")
            tf.data.experimental.save(comb_ds, tf_record_out_file, "GZIP")

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


    def preproc(self):
        import numpy as np
        import sys
        import tensorflow as tf
        sys.path.append('../label/')
        from label import Synthetic_Label

        def generate_beta_coeffs(stack, prev_win, next_win):
            import tensorflow as tf

            return tf.ensure_shape(tf.numpy_function(
                        Synthetic_Label.compute_label_S2_S1_ENDISI_beta_coeefs,
                        [prev_win[3], next_win[3]], tf.float32),
                        [2, 3])

        # Helper for parallel execution...
        def get_coeffs(tile):
            print("{}, {}".format(tile[0], tile[1]))
            sample_file = self.tf_record_path +                                \
                          "{}_{}.tfrecords".format(tile[0], tile[1])
            windows_ds = self._annotate_ds(sample_file)
            coeffs_ds = windows_ds.map(generate_beta_coeffs,
                                       num_parallel_calls=1)

            res = []
            for coeff in coeffs_ds:
                res.append(coeff)
            return np.array(res)


        if not self.generate_labels:
            return None

        num_tiles_y, num_tiles_x = self.get_num_tiles()

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
                                        num_parallel_calls=self.n_threads)

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
            beta1 = Synthetic_Label.compute_label_S2_S1_ENDISI_comp_betas(
                                              all_coeffs[:, window_no, 0, :])
            beta2 = Synthetic_Label.compute_label_S2_S1_ENDISI_comp_betas(
                                              all_coeffs[:, window_no, 1, :])
            window_betas_tmp.append((beta1, beta2))
        return np.array(window_betas_tmp)

    def write_tf_files(self, dst_dir, selector, window_betas = None):
        import os
        import datetime
        import tensorflow as tf

        def write_data(tile):
#            print("{}, {}".format(tile[0], tile[1]))
            sample_file = self.tf_record_path +                                \
                          "{}_{}.tfrecords".format(tile[0], tile[1])
            windows_ds = self._annotate_ds(sample_file)
            if self.generate_labels:
                self._write_training_data(
                                self.tf_record_out_path +                      \
                                dst_dir +                                      \
                                "{}_{}.tfrecords".format(tile[0], tile[1]),    \
                                windows_ds,                                    \
                                window_betas)

            else: # self.generate_labels == False
                self._write_inference_data(
                                self.tf_record_out_path +                      \
                                dst_dir +                                      \
                                "{}_{}.tfrecords".format(tile[0], tile[1]),    \
                                windows_ds)
            return 0


        # Note: If multi-threaded by caller, this test might not work properly.
        #       Hence, we ignore cases where the directory already exists!
        if not os.path.isdir(self.tf_record_out_path + dst_dir):
            try:
                os.mkdir(self.tf_record_out_path + dst_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        num_tiles_y, num_tiles_x = self.get_num_tiles()

#        print("Generating data ({}) for tiles (y, x):".format(dst_dir))
        list_tiles = []
        for j in range(0, num_tiles_y):
            for i in range(0, num_tiles_x):
                if selector(j, i):
                    list_tiles.append((j, i))

        write_ds = tf.data.Dataset.from_generator(lambda: list_tiles,
                                                  output_signature=(
                                                      tf.TensorSpec(
                                                          shape=(2),
                                                          dtype=tf.uint64)))
        # Note: Multi-threading is done externally since writing TFRecord
        #       files is not well parallelizable.
        write_ds = write_ds.map(lambda tile: tf.py_function(
                                        write_data,
                                        [tile],
                                        [tf.uint64]),
                                        num_parallel_calls=1)

        for item in write_ds:
            pass # Just iterate to write the files

        return

    def get_infer_dataset(self, tile):
        import os
        import datetime
        import tensorflow as tf

        sample_file = self.tf_record_path +                                    \
                      "{}_{}.tfrecords".format(tile[0], tile[1])
        if os.path.exists(sample_file):
            return self._annotate_ds(sample_file)

        return None
