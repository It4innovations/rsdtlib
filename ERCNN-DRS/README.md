# Use rsdtlib for ERCNN-DRS

In our previous work of [ERCNN-DRS Urban Change Monitoring](https://github.com/It4innovations/ERCNN-DRS_urban_change_monitoring) the time series processing of (an early version of) `rsdtlib` was used. Two neural network models are trained for the two eras using ERS-1/2 & Landsat 5 TM (1991-2011), and Sentinel 1 & 2 (2017-2021). The required data pre-processing scripts for the two different eras can be found in the subdirectories [`ers12ls5`](./ers12ls5) and [`s12`](./s12), respectively.

## Sentinel 1 & 2
The scripts cover all pre-processing steps from downloading, stacking and windowing of the remote sensing time series observations. For downloading, a Sentinel-Hub account is needed, which then provides a turnkey solution.

Usage example:
    `$ cd ./s12/`
    `$ python download_s12_Liege.py # Add you Sentinel-Hub credentials in the script`
    `$ python stack_s12_Liege.py`
    `$ python window_training_s12_Liege.py`

This set of commands downloads all Sentinel 1 & 2 observations from Sentinel-Hub for the timeframe 2017-01-01 to 2021-01-17. It further builds the windowed training and validation time series data sets that can be used directly for training.

This is available for all three Areas of Interests (AoIs): Rotterdam (Netherlands), Limassol (Cyprus), and Liège (Belgium)

# ERS-1/2 & Landsat 5 TM
For ERS-1/2 & Landsat 5 TM, we provide stacking and windowing scripts. Downloading and product related pre-processing are not coverred as external tools were used (e.g., EarthExplorer Bulk Download Application, ESA SNAP).

Usage example:
    `$ cd ./ers12ls5/`
    `$ python stack_ers12ls5_Liege.py`
    `$ python window_training_ers12ls5_Liege.py`

This expects the observations being pre-processed externally and made availble as `EOPatches`, separately for each product. Also see the [conversion example](../examples/example_convert.py) to help with converting `GeoTIFF` files to `EOPatches`. This set of commands builds the windowed training and validation time series data sets for the time frame 1991-07-01 to 2011-09-01 that can be used directly for training.

This is available for all three Areas of Interests (AoIs): Rotterdam (Netherlands), Limassol (Cyprus), and Liège (Belgium)


## Inference

Both eras contain scripts for inference of the respective AoIs, using the trained models of our previous work. The scripts apply inference to the entire AoI and all windows.

Usage example for Sentinel 1 & 2:
    `$ cd ./s12/`
    `$ python window_interactive_inference_s12_Liege.py`

Usage example for ERS-1/2 & Landsat 5 TM:
    `$ cd ./s12/`
    `$ python window_interactive_inference_ers12ls5_Liege.py`

The results are `NumPy` array files. One for each tile containing the urban change maps of all its windows.

These are available for all three Areas of Interests (AoIs): Rotterdam (Netherlands), Limassol (Cyprus), and Liège (Belgium)
