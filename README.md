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

## Quick Start: clone and compile

Need to do this the first time the repository is cloned from github.

```
git clone https://github.com/larbys/ubdl
cd ubdl
git submodule init
git submodule update
source setenv_py3.sh
source configure.sh
source buidlall_py3.sh
```

## Quick Start: Set up the environment

Need to do this everytime you open a new terminal and want to use the code.

```
(go to top level folder -- the same folder as this README)
source setenv_py3.sh
source configure.sh
```

## Repositories

* [LArCV](https://github.com/larbys/larcv): image format IO library. Version 1 (as opposed to incompatible LArCV2 library).
  `ubdl_dev` branch which is quite different from LArCV1 version used in past chain.  Inroduces much of the changes in LArCV2.
  But does maintain some backwards-compatibility -- primarily reading of old-LArCV1 input.
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
| setenv_py3.sh    | For python3 build. Set envionroment variables for packages and libraries not included in this repository (e.g. ROOT, opencv, pytorch) | YES (machine dependent) | every new shell |
| setenv_fnal.sh    | set envionroment variables for external packages and libraries ON FNAL (e.g. ROOT, opencv, pytorch) | NO | every new shell |
| configure.sh | setup the environment variables of packages included in `ubdl`. | NO | every new shell |
| buildall_py2.sh  | run commands to build all the modules inside this repo | NO (but call to compile) | NO | every time changes are made to source. and after first cloning the repo |
| buildall_py2.sh  | For python3 build. Run commands to build all the modules inside this repo | NO (but call to compile) | NO | every time changes are made to source. and after first cloning the repo |
| cleanall.sh  | run `make clean` in all repo folders | NO | NO | when you want to hit the reset button |
| first_setup.sh | loads all of the gitsubmodules. call this the first time you clone the repository | NO | after the first time cloning the repo |
| tufts_start_container.sh | Setup for python 2 build | NO | On Tufts. |
| tufts_start_container_py3.sh | Setup for python 2 build | NO | On Tufts. |
| tufts_submit_build.sh | For python 2 build on Tufts | You should check the container you build with | On Tufts. |
| tufts_submit_build_python3.sh | For python 3 build on Tufts | You should check the container you build with | On Tufts. |


## scripts directory

| script                   | purpose | do I have to edit it? | when do I run it? |
|:------------------------:|:-------:|:---------------------:| :---------------- |
| setenv_ublarcvserver.sh  | set envionroment variables for using ublarcvserver | NO | every new shell |
| start_ublarcvserver_broker.sh | start the gpu job broker | NO | when you want to start a broker |
| start_ublarcvserver_worker.sh | start one or more gpu UBSSNet workers | NO | when you want to start workers |
| run_ubssnet_client.sh | run ubssnet client | NO | when you want to run a client |
| tufts_start_container.sh | start current ubdl container stored on Tufts cluster | NO | to start a new interactive session inside the container |
| tufts_submit_build.sh    | launches a grid job on the Tufts cluster to build code | NO (but call to compile) | every time changes are made to source code |


## LArCV1 notes

More details about the LArCV1 version used here.

* to do

## TO DO LIST

* go back reconfigure build system to use cmake: geo2d, laropencv?
* support a build mode that produces a folder for UPS product
* documentation for how to setup the different net's workers
