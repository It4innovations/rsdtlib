Workflow of `rsdtlib`
=====================

The purpose of `rsdtlib` is to provide an end-to-end solution for three processing stages to prepare a time series of multi-modal remote sensing observations:

    - | Stage 1:
      | Download remote sensing data directly from Sentinel Hub (i.e. Sentinel 1 & 2), or convert existing `GeoTIFF` files
    - | Stage 2:
      | Temporally stack, assemble, and tile these observations
    - | Stage 3:
      | Create windows of longer time series comprising these observations (i.e. deep-temporal)

Below figure shows the processing pipeline considering all three stages with data formats in orange.

.. figure:: ../images/rsdtlib_pipeline.png

In the following, stages 2 and 3 are detailed with parameters to control the processes.

.. figure:: ../images/temporal_stacking_windowing.png

