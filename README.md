# MLSI
MLSI is short for Maschine Learning Structure Identifier. A neural network based crystal structure identification engine implemented as python script modifier for OVITO.

The model used here is build on a [PyTorch](https://pytorch.org/) implementation for "[Dynamic Graph CNN for Learning on Point Clouds (DGCNN)](https://arxiv.org/pdf/1801.07829)" by [An Tao](antao97/dgcnn.pytorch).

There are two different pretrained models provided in this repo.
1. `Models/Simple_Crystal_Structures` is trained to classify different simple crystal structures (BCC, cubic diamond, cubic perovskite, FCC, HCP) and their melts. This structure identifier was mainly used for comparison with established methods.
2. `Models/SiO2_polymorphs` is trained to differentiate a set of different silica polymorphs. Currently: Coesite 1, "Cristobalite 1", "Cristobalite 2", "Quartz 1", "Quartz 2", "Tridymite 1", "Tridymite 2", "Moganite 1", "Moganite 2", and "Stishovite 1" are supported. Moreover, the liquid phase can be seperated as "Melt".

Please note: This repo and all its content are still in beta. Proceed at your own discretion and please contact me about bugs etc.
# Installation
[PyTorch](https://pytorch.org/) can be installed into any existing [OVITO Pro](https://www.ovito.org/) version using the following command: `ovitos -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu116` (depending on the desired PyTorch version).

In a similar fashion, [SciPy](https://scipy.org/) may be installed with: `ovitos -m pip install scipy`.

# Usage
1. Load your structure into OVITO
2. Insert a `Python Script Modifer` into the pipeline and load the `OVITO_MLSI.py` script.
3. You will be presented a set of options in the sidebar:
    - `model_dir`: The path to the model and settings files, e.g., `OVITO_MLSI/Models/SiO2_polymorphs`
    - `cutoff`: Score value [0,1] below which an atom is classified as type "Other".
    - `NearestNeighborFinder` / `KDTree` toggle. These switches select the method of building neighbor lists. `NearestNeighborFinder` should be used whenever possible (i.e. sufficiently small neighbor count in the model). `KDTree` works form much greater neighborhood sizes, however, it has only a "manual" periodic boundary conditions in the form of inserted padding atoms. 
    - The `rep` and `buffer` paramters control this padding layer. `rep` gives the number of inserted simulation cells in each periodic direction and `buffer` controls how many of these atoms are kept during classification. Example: `rep = 3` and `buffer = 0.1` implies that each box dimension is expanded by 10% during the classification step so that each atom has a sufficient number of neighbors.

# Requirements
The code provided here most likely works on many versions of [PyTorch](https://pytorch.org/), [OVITO Pro](https://www.ovito.org/), and [SciPy](https://scipy.org/). However, our testing environment consists of the following versions:
- OVITO Pro 3.7.9
- PyTorch 1.12.1+cu116 or 1.12.1+cpu 
- Ubuntu 20.04 LTS.
- The KDTree version of the neigbor finder was tested using SciPy version 1.9.1.
- For CUDA, NVIDIA driver 510.85.02 with CUDA Version 11.6 was used.

# Disclaimer
This project and any code therein is not associated or affiliated to the OVITO GmbH.

# Acknowledgement
Work on this project was funded by the National High Performance Computing Center for Computational Engineering Science [NHR4CES](https://www.nhr4ces.de/).
