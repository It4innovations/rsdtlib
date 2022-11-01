# rsdtlib: Remote Sensing with Deep-Temporal Data Library

This project provides a Python module that allows:
- **Stage 1:** Download remote sensing data directly from [Sentinel Hub](https://www.sentinel-hub.com/) (i.e. Sentinel 1 & 2)
- **Stage 2:** Temporally stack, assemble and tile these observations
- **Stage 3:** Create windows of longer time series comprising these observations (i.e. deep-temporal)

Below figure shows the processing pipeline considering all three stages:
<p align="center">
  <img src="./images/rsdtlib_pipeline.png" />
</p>

The processing in stages 2 and 3 is detailed below, showing how observations are combined to produce the final windows:
<p align="center">
  <img src="./images/temporal_stacking_windowing.png" />
</p>

# Paper and Citation
TBD

# Contact
Should you have any feedback or questions, please contact the main author: Georg Zitzlsberger (georg.zitzlsberger(a)vsb.cz).

# Acknowledgments
This research was funded by ESA via the Blockchain ENabled DEep Learning for Space Data (BLENDED) project (SpaceApps Subcontract No. 4000129481/19/I-IT4I) and by the Ministry of Education, Youth and Sports from the National Programme of Sustainability (NPS II) project “IT4Innovations excellence in science - LQ1602” and by the IT4Innovations Infrastructure, which is supported by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90140) via the Open Access Grant Competition (OPEN-25-24).

The authors would like to thank ESA for funding the study as part of the BLENDED project<sup>1</sup> and IT4Innovations for funding the compute resources via the Open Access Grant Competition (OPEN-25-24). Furthermore, the authors would like to thank the data providers (USGS, ESA, Sentinel Hub and Google) for making remote sensing data freely available:
- Landsat 5 TM courtesy of the U.S. Geological Survey.
- ERS-1/2 data provided by the European Space Agency.
- Contains modified Copernicus Sentinel data 2017-2021 processed by Sentinel Hub (Sentinel 1 & 2).

The authors would finally like to thank the BLENDED project partners for supporting our work as a case study of the developed platform.

<sup>1</sup> [Valentin, B.; Gale, L.; Boulahya, H.; Charalampopoulou, B.; Christos K., C.; Poursanidis, D.; Chrysoulakis, N.; Svato&#x0148;, V.; Zitzlsberger, G.; Podhoranyi, M.; Kol&#x00E1;&#x0159;, D.; Vesel&#x00FD;, V.; Lichtner, O.; Koutensk&#x00FD;, M.; Reg&#x00E9;ciov&#x00E1;, D.; M&#x00FA;&#x010D;ka, M. BLENDED - USING BLOCKCHAIN AND DEEP LEARNING FOR SPACE DATA PROCESSING. Proceedings of the 2021 conference on Big Data from Space; Soille, P.; Loekken, S.; Albani, S., Eds. Publications Office of the European Union, 2021, JRC125131, pp. 97-100.  doi:10.2760/125905.](https://op.europa.eu/en/publication-detail/-/publication/ac7c57e5-b787-11eb-8aca-01aa75ed71a1)

# License
This project is made available under the GNU General Public License, version 3 (GPLv3).
