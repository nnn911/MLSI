# MLSI
MLSI is short for Maschine Learning Structure Identifier. A neural network based crystal structure identification engine implemented as python modifier modifier for OVITO.

The model used here is build on a [PyTorch](https://pytorch.org/) implementation for "[Dynamic Graph CNN for Learning on Point Clouds (DGCNN)](https://arxiv.org/pdf/1801.07829)" by [An Tao](antao97/dgcnn.pytorch).

There are two different pretrained models provided in this repo.
1. `Models/Simple_Crystal_Structures` is trained to classify different simple crystal structures (BCC, cubic diamond, cubic perovskite, FCC, HCP) and their melts. This structure identifier was mainly used for comparison with established methods.
2. `Models/SiO2_polymorphs` is trained to differentiate a set of different silica polymorphs. Currently: Coesite 1, "Cristobalite 1", "Cristobalite 2", "Quartz 1", "Quartz 2", "Tridymite 1", "Tridymite 2", "Moganite 1", "Moganite 2", and "Stishovite 1" are supported. Moreover, the liquid phase can be seperated as "Melt".

When you install the modifier, these will be placed in the MLSI/pretrained_models folder. This folder can resolved dynamically using python's `importlib.resources.files('MLSI.pretrained_models')`.

This repo is in development. Please create issues for issues or bugs you encounter.

# Installation

- OVITO Pro [integrated Python interpreter](https://docs.ovito.org/python/introduction/installation.html#ovito-pro-integrated-interpreter):
  ```
  ovitos -m pip install --user git+https://github.com/nnn911/MLSI.git
  ``` 
  The `--user` option is recommended and [installs the package in the user's site directory](https://pip.pypa.io/en/stable/user_guide/#user-installs).

- Other Python interpreters or Conda environments:
  ```
  pip install git+https://github.com/nnn911/MLSI.git
  ```

This will install all dependencies. Depending on your system, you may need to install [PyTorch](https://pytorch.org/) manually / beforehand to make use of your GPU.

# Usage

1. Load your structure into OVITO
2. Insert a `Python Script Modifer` into the pipeline and load the `OVITO_MLSI.py` script.
3. You will be presented a set of options in the sidebar:
    - *Model directory* `model_dir`: The path to the model and settings files, e.g., `MLSI/pretrained_models/SiO2_polymorphs`
    - *Cutoff* `cutoff`: Score value [0,1] below which an atom is classified as type "Other".
    - *Use: Nearest Neighbor Finder* `use_neighbor_finder` / *Use: KDTree* `use_KDTree` toggle. These switches select the method of building neighbor lists. *Nearest Neighbor Finder* should be used whenever possible (i.e. sufficiently small neighbor count in the model). *KDTree* works form much greater neighborhood sizes, however, it has only "manual" periodic boundary conditions in the form of inserted padding atoms. 
    - *Replicate cell* `rep` and *Thickness of buffer zone* `buffer`: Parameters control this padding layer. *Replicate cell* gives the number of inserted simulation cells in each periodic direction and *Thickness of buffer zone* controls how many of these atoms are kept during classification. Example: `rep = 3` and `buffer = 0.1` implies that each box dimension is expanded by 10% during the classification step so that each atom has a sufficient number of neighbors.

# Requirements

The code provided here most likely works on many versions of [PyTorch](https://pytorch.org/), [OVITO Pro](https://www.ovito.org/), and [SciPy](https://scipy.org/). However, our testing environment consists of the following versions:
- OVITO Pro 3.10.1
- PyTorch 1.13.1
- MacOS
- The KDTree version of the neigbor finder was tested using SciPy version 1.9.1.

# Disclaimer

This project and any code therein is not associated or affiliated to the OVITO GmbH. However, one of the authors (DU) is an employee of the OVITO GmbH.

# Acknowledgement

Work on this project was funded by the National High Performance Computing Center for Computational Engineering Science [NHR4CES](https://www.nhr4ces.de/).
