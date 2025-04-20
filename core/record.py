"""Record.

A record is a collection of measurement from different sensors
(lidar, radar, imu, etc.)
"""
from glob import glob
import sys
import os
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from core.config import ROOTDIR, DATASET
from core.lidar import Velodyne
from core.radar import CCRadar
from core.radar import MCCRadar

from .utils.common import error

from typing import List, Dict, Tuple

from tqdm import tqdm
class Record:
    """Record.

    Class describing records in the dataset

    Attributes:
        calibration: Calibration parameters
        lidar: Velodyne samples of the dataset record
    """

    def __init__(self, descriptor: Dict[str, Dict[str, str]],
                 calibration, codename: str, index: int) -> None:
        """Init.

        Arguments:
            descriptor: Holds the paths that describe the dataset
            calibration: Calibration object
                Provide all the calibration parameters of all sensors
            codename: Subdirectory codename
            index: Order number indicating the entry of the dataset in interest
        """
        self.calibration = calibration
        self.descriptor = descriptor
        self.index = index
        self.codename: str = codename
        records_dict = {
            "folders": "path",
            "superfolders": "subcodenames"
        }
        combined_records, ind2key = self._combine_records(records_dict)
        sub_ind: str = ""
        for i, dataset in enumerate(combined_records):
            if dataset["codename"] == codename:
                key = ind2key(i)
                sub_ind = dataset[key]
                cmobined_ind = i
                break
        if not sub_ind:
            error(f"Dataset codename '{codename}' not defined in '{DATASET}")
            sys.exit(1)
        if key == "path": self.descriptor["paths"]["rootdir"] = os.path.join(ROOTDIR, sub_ind)
        elif key == "subcodenames":
            self.descriptor["paths"]["rootdir"] = [os.path.join(ROOTDIR, ind) for ind in sub_ind]
            self.descriptor["codename"] = self.codename
            self.descriptor["subcodenames"] = sub_ind
            self.descriptor["poses"] = combined_records[cmobined_ind]["poses"]
        self.lidar = None
        self.ccradar = None
        # self.mccradar = None

    def _combine_records(self, records_dict):
        """Combine records from multiple lists into a single list."""
        combined = []
        ind_dict = {}
        ind = 0
        for k,v in records_dict.items():
        # for lst in [self.descriptor[key] for key in records_dict]:
            lst = self.descriptor[k]
            combined.extend(lst)
            ind_dict.update({(ind, ind+len(lst)-1): v})
            ind += len(lst)
        # Create a mapping from original index to new index
        def ind2key(idx):
            for key, value in ind_dict.items():
                if idx >= key[0] and idx <= key[1]:
                    return value
            return None
        return combined, ind2key

    def load(self, sensor: str) -> None:
        """Load the data file for a given sensor.

        Arguments:
            sensor: The sensor considered so that only the data of that sensor
                    would be loaded.
                    Possible Values: lidar, scradar, ccradar
        """

        if sensor == "velodyne":
            self.lidar = Velodyne(self.descriptor, self.calibration, self.index)
        elif sensor == "ccradar":
            self.calibration.ccradar.load_waveform_config(self.descriptor)
            self.ccradar = CCRadar(self.descriptor, self.calibration, self.index)
            # self.ccradar.adc_bckg = np.load("/mnt/storage/DATA/SpecialOps/rwu-dataset/dataset/MMWL_Capture_1743691122/outputs/MMWL_Capture_1743691122/ccradar/average_adcSpRw.npy")
            # self.ccradar.sig_bckg = np.load("/mnt/storage/DATA/SpecialOps/rwu-dataset/dataset/MMWL_Capture_1743931222/outputs/MMWL_Capture_1743931222/ccradar/average_sigPow.npy")
        elif sensor == "mccradar":
            self.calibration.mccradar.load_waveform_config(self.descriptor)
            self.ccradar = MCCRadar(self.descriptor, self.calibration, self.index)

    def process_and_save(self, sensor: str, **kwargs) -> None:
        """Process and save the result into an output folder.

        Arguments:
            sensor (str): Name of sensor of interest
                          Values: "lidar", "scradar", "ccradar"
            kwargs (dict): Keyword argument
                "threshold": Threshold value to be used for rendering
                             radar heatmap
                "no_sidelobe": Ignore closest recording in each frame
                "velocity_view": Enable the rendering of radial velocity
                                 as fourth dimention
                "heatmap_3d": Save 3D heatmap when true. Otherwise, a
                              2D heatmap is generated
                "save_as": Save arrays as `.csv` or `.bin` files
                "index": Start index
        """
        # Dot per inch
        self._dpi: int = 400
        self._kwargs = kwargs
        self._sensor = sensor
        start_idx: int = kwargs.get("start_index", 1)

        # Output directory path
        output_dir: str = kwargs.get("output", "output")
        output_dir = f"{output_dir}/{self.codename}/{sensor}"
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        cpu_count: int = multiprocessing.cpu_count()
        print(f"Please wait! Processing on {cpu_count} CPU(s)")

        if sensor == "velodyne":
            dataset_path: str = os.path.join(
                self.descriptor["paths"]["rootdir"],
                self.descriptor["paths"][sensor]["pointcloud"]["data"]
            )
            nb_files: int = len(os.listdir(dataset_path)) - 1
            with multiprocessing.Pool(cpu_count) as pool:
                pool.map(
                    self._process_lidar,
                    range(start_idx, nb_files + 1),
                    chunksize=10
                )
        elif (sensor == "ccradar") or (sensor == "scradar"):
            dataset_path: str = os.path.join(
                self.descriptor["paths"]["rootdir"],
                self.descriptor["paths"][sensor]["raw"]["data"]
            )
            nb_files: int = len(os.listdir(dataset_path)) - 1
            # results = []
            with multiprocessing.Pool(cpu_count) as pool:
                pool.map(
                    self._process_radar,
                    range(start_idx, nb_files + 1),
                    chunksize=10
                )
            
            # self.save_results(results)
        
        elif sensor == "mccradar":
            self._kwargs["txl"] = self.calibration.ccradar.antenna.txl
            self._kwargs["rxl"] = self.calibration.ccradar.antenna.rxl
            for i in tqdm(range(start_idx, 20)):
                self._process_radar(i)
            # nb_files = 40
            # with multiprocessing.Pool(cpu_count) as pool:
            #     pool.map(
            #         self._process_radar,
            #         range(start_idx, nb_files + 1),
            #         chunksize=10
            #     )
    
    # def save_results(self, results: List[int]) -> None:
    #     for res in results:
    #         if res == -1:
    #             print("Error in processing")
    #             continue
    #         signal_power, adc_samples_raw = res[0], res[1]
    #         print(
    #             f"[ ========= {100 * res/len(results): 2.2f}% ========= ]\r",
    #             end=""
    #         )
    
    def _process_radar(self, idx: int) -> int:
        """Handler of radar data processing.

        Used as the handler for parallel processing. The context attributes
        needed by this method are only defined in the method `process_and_save`
        As so, only that method is supposed to call this one.

        NOTE: THIS METHOD IS NOT EXPECTED TO BE CALLED FROM OUTSIDE OF THIS
        CLASS

        Argument:
            idx: Index of the file to process
        """
        self.index = idx
        self.load(self._sensor)
        SIZE: int = 18   # inch
        plt.figure(1, clear=True, dpi=self._dpi, figsize=(SIZE, SIZE))
        if self._kwargs.get("heatmap_3d") == False:
            self.ccradar.show2dHeatmap(False, False)
        elif self._kwargs.get("heatmap_3d"):
            self.ccradar.showHeatmapFromRaw(
                self._kwargs.get("threshold"),
                self._kwargs.get("no_sidelobe"),
                self._kwargs.get("velocity_view"),
                self._kwargs.get("polar"),
                show=False,
            )
        elif self._kwargs.get("pointcloud"):
            if self._kwargs.get("save_as") == "csv":
                pcl = self.ccradar.getPointcloudFromRaw(
                    polar=self._kwargs.get("polar")
                )
                np.savetxt(
                    f"{self._output_dir}/radar_pcl{idx}.csv",
                    pcl.astype(np.float32),
                    delimiter=",",
                    header="Azimuth (m), "
                           "Range (m), "
                           "Elevation (m), "
                           "Velocity (m/s), "
                           "Intensity or SNR (dB)"
                )
                return idx
            elif self._kwargs.get("save_as") == "bin":
                pcl = self.ccradar.getPointcloudFromRaw(
                    polar=self._kwargs.get("polar"))
                pcl.astype(np.float32).tofile(
                    f"{self._output_dir}/radar_pcl{idx}.bin")
                return idx
            
            self.ccradar.showPointcloudFromRaw(
                self._kwargs.get("velocity_view"),
                self._kwargs.get("bird_eye_view"),
                self._kwargs.get("polar"),
                camera_view=self._kwargs.get("camera_view"),
                show=False,
            )
            
        plt.savefig(f"{self._output_dir}/radar_{idx:04}_2dd.jpg", dpi=self._dpi)
        # plt.savefig(f"{self._output_dir}/radar_{idx:04}_pcl.jpg", dpi=self._dpi)
        # plt.savefig(f"{self._output_dir}/radar_{idx:04}_adcsprw_avg_sub0.jpg", dpi=self._dpi)
        # plt.savefig(f"{self._output_dir}/radar_{idx:04}_sigpow_avg_sub.jpg", dpi=self._dpi)

        # signal_power, adc_samples_raw = res[0], res[1]
        # np.save(f"{self._output_dir}/radar_{idx:04}_sigPow.npy", signal_power)
        # np.save(f"{self._output_dir}/radar_{idx:04}_adcSpRw.npy", adc_samples_raw)
        return idx

    def _process_lidar(self, idx: int) -> int:
        """Handler of lidar data processing.

        Used as the handler for parallel processing. The context attributes
        needed by this method are only defined in the method `process_and_save`
        As so, only that method is supposed to call this one.

        NOTE: THIS METHOD IS NOT EXPECTED TO BE CALLED FROM OUTSIDE OF THIS
        CLASS

        Argument:
            idx: Index of the file to process
        """
        self.index = idx
        self.load(self._sensor)
        bev = self.lidar.getBirdEyeView(
            self._kwargs.get("resolution", 0.05),
            self._kwargs.get("srange"),
            self._kwargs.get("frange"),
        )
        plt.imsave(f"{self._output_dir}/lidar_bev_{idx:04}.jpg", bev)

    def make_video(self, inputdir: str, ext: str = "jpg", prext = "") -> None:
        """Make video out of pictures"""
        # prext = "_pcl"#"_pcl" or "_2d"
        files = glob(inputdir + f"/*{prext}.{ext}")
        files = sorted(files)
        height, width, _ = plt.imread(files[0]).shape
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        video = cv.VideoWriter(inputdir + f"/{self.codename}{prext}.avi", fourcc, 10, (width, height))
        for idx, img in enumerate(files):
            print(
                f"[ ========= {100 * idx/len(files): 2.2f}% ========= ]\r",
                end=""
            )
            video.write(cv.imread(img))
        cv.destroyAllWindows()
        video.release()
