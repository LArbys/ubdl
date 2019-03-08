# ubdl

Collection of repositories for the (new) MicroBooNE DL Reconstruction Chain

## Repositories

* [LArCV](https://github.com/larbys/larcv): image format IO library. Version 1 (as opposed to incompatible LArCV2 library).
  `ubdl_dev` branch which is quite different from LArCV1 version used in past chain.  Inroduces much of the chains in LArCV2.
  But does maintain some backwards-compatibility (just reading of old-LArCV1 input).
* [larlite](https://github.com/larlight/larlite): analysis framework for MicroBooNE. Uses `dlreco_larflow_merge` branch.
* [Geo2D](https://github.com/LArbys/Geo2D): 2D geometry tools (based on OpenCV)
* [LArOpenCV](https://github.com/NevisUB/LArOpenCV): pattern recognition for 1l1p neutrino vertices. Uses OpenCV algorithms.
* [ublarcvapp](https://github.com/larbys/ublarcvapp): applications built on larcv (and the other libraries).
* [ublarcvserver](https://github.com/larbys/ublarcvserver): tools for processing of network data via remote gpus

# Setting up the Environment

There are a lot of packages which need their environment variables setup when running.

We provide a number of scripts:

| script       | purpose | do you have to edit it? |
|:-----------: |:-------:|:------------------------|
| setenv.sh    | set envionroment variables for packages and libraries not included in this repository (e.g. ROOT, opencv, pytorch) | YES (machine dependent) |
| configure.sh | setup the environment variables of packages included in `ubdl`. | NO |
| buildall.sh  | run commands to build all the modules inside this repo | NO (but call to compile) | NO |
| first_setup.sh | loads all of the gitsubmodules. call this the first time you clone the repository | NO |


## LArCV1 notes

More details about the LArCV1 version used here.

* to do