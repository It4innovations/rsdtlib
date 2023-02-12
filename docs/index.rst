.. rsdtlib documentation master file, created by
   sphinx-quickstart on Thu Feb  9 15:53:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

This is the documentation of `rsdtlib` found on `GitHub <https://github.com/It4innovations/rsdtlib/>`_

Workflow of `rsdtlib`
=====================

The purpose of `rsdtlib` is to provide an end-to-end solution for three processing stages to prepare a time series of multi-modal remote sensing observations:

    - | Stage 1:
      | Download remote sensing data directly from Sentinel Hub (i.e. Sentinel 1 & 2), or convert existing `GeoTIFF` files
    - | Stage 2:
      | Temporally stack, assemble and tile these observations
    - | Stage 3:
      | Create windows of longer time series comprising these observations (i.e. deep-temporal)

Below figure shows the processing pipeline considering all three stages with data formats in orange:

.. figure:: ../images/rsdtlib_pipeline.png

Detailed stages 2 and 3 with parameters to control the processes:

.. figure:: ../images/temporal_stacking_windowing.png

API of `rsdtlib`
================

.. automodule:: rsdtlib
    :members:
