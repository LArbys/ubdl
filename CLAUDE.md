# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

UBDL (MicroBooNE Deep Learning) is a neutrino physics reconstruction framework that implements the "Gen-2" workflow for analyzing Liquid Argon Time Projection Chamber (LArTPC) data using deep convolutional neural networks. It reconstructs 3D neutrino interactions from 2D wire-plane images through a sophisticated 7-step pipeline.

## Core Build System

### Initial Setup (First Time)
```bash
git submodule init && git submodule update
source setenv_py3_container.sh
source configure_container.sh
source buildall_py3.sh
```

### Daily Development
```bash
source setenv_py3_container.sh
source configure_container.sh
```

### Building Individual Components
```bash
cd [component_name]
make install -j4
```

### Clean Build
```bash
source cleanall.sh  # Clean all modules
source buildall_py3.sh  # Full rebuild
```

## Environment Configuration

The codebase uses machine-specific environment setup:
- `setenv_py3_container.sh`: Configures external dependencies (ROOT, CUDA, OpenCV, PyTorch)
- `configure_container.sh`: Sets up UBDL component environment variables
- Environment automatically detects container vs. native machine setup

## Core Architecture

### Component Structure (Git Submodules)
- **larcv/**: Image data format and I/O (Image2D, ROOT integration)
- **larlite/**: Lightweight physics analysis framework 
- **larflow/**: 3D spacepoint reconstruction with LArMatch neural networks
- **ublarcvapp/**: MicroBooNE-specific applications and detector calibrations
- **Geo2D/**: 2D geometry utilities (OpenCV-based)
- **LArOpenCV/**: Computer vision algorithms for vertex finding
- **lardly/**: Interactive 3D event visualization (Plotly/Dash)
- **cilantro/**: 3D point cloud processing

### Neural Network Components
- **LArMatchNet**: Sparse 3D CNN using MinkowskiEngine for pixel correspondence
- **Shower Reconstruction**: Graph neural networks in `larflow/dlshowermodel/`
- **KeyPoint Detection**: Physics vertex/endpoint identification
- **Infill Networks**: Complete missing detector data

## Development Workflows

### LArMatch Network Development
```bash
cd larflow/larmatchnet
source set_pythonpath.sh
python train_dist_larmatchme.py --config config.yaml
```

### Container Development
Primary development occurs in Singularity containers with pre-built dependencies (ROOT, PyTorch, MinkowskiEngine, CUDA).

### Testing
- Component tests: Each module has test directories with Python scripts
- Integration tests: End-to-end reconstruction pipeline validation
- Physics validation: Reconstruction performance on simulated data
- Tutorial system: `Tutorials/` directory with Jupyter notebooks

## Current Branch Context

**Branch**: `lantern_v3dev_showerkp_retraining`
- Focus: Shower reconstruction retraining with keypoint improvements
- Recent work: LArMatch network retuning, cosmic ray tagging integration

## Key Technologies

- **PyTorch**: Deep learning framework with CUDA support
- **MinkowskiEngine**: Sparse 3D convolution for point clouds
- **ROOT**: Physics data analysis and I/O
- **OpenCV**: Computer vision algorithms
- **XGBoost**: Boosted decision trees for particle selection
- **HDF5**: Training dataset storage

## Data Pipeline

1. **LArTPC wireplane images** → 2. **Infill network** → 3. **LArFlow pixel matching** → 4. **Clustering** → 5. **Light matching** → 6. **Particle ID** → 7. **Interaction graphs**

## Development Commands

### Build Commands
- `source buildall_py3.sh`: Build all modules
- `cd [module] && make install`: Build single module
- `source cleanall.sh`: Clean all builds

### No Standard Test Runner
Tests are module-specific Python scripts. Check individual component directories for test files and run them directly with Python.

### Environment Commands  
- `source setenv_py3.sh && source configure.sh`: Set up environment (run daily)
- Machine detection is automatic (container vs. native)

## File Patterns

- Neural network configs: `*.yaml`, `*.cfg` files
- Training scripts: `train_*.py`, `deploy_*.py`
- Data processing: `*_dataprep.py`, `prep*.py`
- Visualization: `view_*.py`, `test_*.py`
- ROOT macros: `*.cxx`, `*.C`

## Special Considerations

- **Container Environment**: Most development in Singularity containers
- **GPU Requirements**: Networks require 8-32GB GPU memory
- **Large Datasets**: Training on TB-scale physics simulation data
- **Distributed Training**: PyTorch DistributedDataParallel support
- **Physics Domain**: Neutrino detection and particle physics reconstruction