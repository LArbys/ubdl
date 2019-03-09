# ubdl

Collection of repositories for the (new) MicroBooNE DL Reconstruction Chain

The vision for this reconstruction chain is to

1) take in LArTPC image data (post noise filtering and wire signal deconvolution)
2) run the `infill` network to propose tracks across dead regions
3) run `larflow` to take 2D images from all three planes and produce 3D spacepoints
4) run `sametime cluster` network to cluster charge (in 2D) that was created as part of the same particle cascade (e.g. cosmics, beam neutrinos)
5) form hypothesis for the light signal seen by the PMTs by the cluster. use the hypotheses to choose clusters consistent with the in-time flash
6) for the chosen clusters, cluster the individual particles
7) determine the relationship between particles by proposing an interaction graph

All these steps use deep convolutional neural networks.

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
There are also a number of packages to build (a few of which do not have CMake config files).

We provide a number of scripts to set the environment variables and perform the build:

| script       | purpose | do I have to edit it? | when do I run it? |
|:-----------: |:-------:|:---------------------:| :---------------- |
| setenv.sh    | set envionroment variables for packages and libraries not included in this repository (e.g. ROOT, opencv, pytorch) | YES (machine dependent) | every new shell |
| configure.sh | setup the environment variables of packages included in `ubdl`. | NO | every new shell |
| buildall.sh  | run commands to build all the modules inside this repo | NO (but call to compile) | NO | every time changes are made to source. and after first cloning the repo |
| first_setup.sh | loads all of the gitsubmodules. call this the first time you clone the repository | NO | after the first time cloning the repo |



## scripts directory


## LArCV1 notes

More details about the LArCV1 version used here.

* to do