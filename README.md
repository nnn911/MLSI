# MLSI
MLSI is short for Maschine Learning Structure Identifier. A neural network based crystal structure identification engine implemented as python modifier modifier for OVITO.

The model used here is build on a [PyTorch](https://pytorch.org/) implementation for "[Dynamic Graph CNN for Learning on Point Clouds (DGCNN)](https://arxiv.org/pdf/1801.07829)" by [An Tao](antao97/dgcnn.pytorch).

There are two different types of pretrained models provided in this repo.
1. `pretrained_models/Simple_Crystal_Structures` is trained to classify different simple crystal structures (BCC, cubic diamond, cubic perovskite, FCC, HCP) and their melts. This structure identifier was mainly used for comparison with established methods. We provide structure identifiers trained with artificial training data as well trained with MD training data. 
2. `pretrained_models/SiO2_polymorphs` is trained to differentiate a set of 25 different silica phases. These phases include alpha-quartz, beta-quartz, low temperature tridymite, high temperature tridymite, alpha cristobalite, beta cristobalite, alpha moganite, beta moganite, coesite, stishovite, CaCl2-type, pyrite-type, seiferite, rosiaite, d-NiAs-type, SnO2-type, NaTiF4-type and some other high-pressure phase, as well as the amorphous phase and the melt.

When you install the modifier, these will be placed in the MLSI/pretrained_models folder. This folder can resolved dynamically using python's `importlib.resources.files('MLSI.pretrained_models')`.

This repo is in development. Please create issues for issues or bugs you encounter.

# Installation

- OVITO Pro [integrated Python interpreter](https://docs.ovito.org/python/introduction/installation.html#ovito-pro-integrated-interpreter):
  ```
  ovitos -m pip install --user git+https://github.com/nnn911/MLSI.git
  ``` 
  The `--user` option is recommended and [installs the package in the user's site directory](https://pip.pypa.io/en/stable/user_guide/#user-installs).

- Other Python interpreters:
  ```
  pip install git+https://github.com/nnn911/MLSI.git
  ```

- If you use conda you need to install OVITO first via conda.

  ```
  conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.10.1
  ```

  Then everything else can be installed via pip.

  ```
  pip install git+https://github.com/nnn911/MLSI.git
  ```

This will install all dependencies. Depending on your system, you may need to install [PyTorch](https://pytorch.org/) manually / beforehand to make use of your GPU.

# Usage

1. Load your structure into OVITO
2. You should be able to directly select the 'MLSI' Modifier in the 'Add modification...' tool bar.
3. You will be presented a set of options in the sidebar:
    - *Model directory* `model_dir`: The path to the model and settings files, e.g., `MLSI/pretrained_models/SiO2_polymorphs`
    - *Cutoff* `cutoff`: Score value [0,1] below which an atom is classified as type "Other".
    - *Use: Nearest Neighbor Finder* `use_neighbor_finder` / *Use: KDTree* `use_KDTree` toggle. These switches select the method of building neighbor lists. *Nearest Neighbor Finder* should be used whenever possible (i.e. sufficiently small neighbor count in the model). *KDTree* works form much greater neighborhood sizes, however, it has only "manual" periodic boundary conditions in the form of inserted padding atoms. 
    - *Replicate cell* `rep` and *Thickness of buffer zone* `buffer`: Parameters control this padding layer. *Replicate cell* gives the number of inserted simulation cells in each periodic direction and *Thickness of buffer zone* controls how many of these atoms are kept during classification. Example: `rep = 3` and `buffer = 0.1` implies that each box dimension is expanded by 10% during the classification step so that each atom has a sufficient number of neighbors.

Alternatively, you can use the modifier directly within the python for example with the following lines:

  ```
  from ovito.io import *
  from MLSI import *

  pipeline.modifiers.append(MLSI(
    model_dir = 'MLSI/pretrained_models/SiO2_polymorphs', 
    cutoff = 0.5, 
    buffer = 0.0))

  data = pipeline.compute()
  export_file(data,"dump.xyz","xyz", columns=["Particle Identifier","Particle Type", "Position.X", "Position.Y","Position.Z","Structure","Score"])

  ```


# Requirements

The code provided here most likely works on many versions of [PyTorch](https://pytorch.org/), [OVITO Pro](https://www.ovito.org/), and [SciPy](https://scipy.org/). However, our testing environment consists of the following versions:
- OVITO Pro 3.10.1
- PyTorch 1.13.1
- MacOS
- The KDTree version of the neigbor finder was tested using SciPy version 1.9.1.



# Fitting your own model

Please use the scripts provided in MLSI/fitting_models to create your own structure identification models. To run these scripts you need to install additionally the following packages:

- tqdm

- h5py

- shutil

- sched

- tempfile

These can be installed with

```
  pip install --user tqdm h5py shutil schedule temp
```

or 

```
  ovitos -m pip install -user tqdm h5py shutil schedule temp
``` 

To perform a fit create an empty directory. Place the scripts from fitting_models within this directory and create a sub folder md_training_data. Please create now for each phase its own sub folder within the md_training_data folder and place the MD snapshots in a OVITO readable in these folders (see https://www.ovito.org/docs/current/python/modules/ovito_io.html). Your folder structure should looks something like this

- Structure_identification
    - DataLoader.py
    - extract_data_ovito.py
    - Model.py
    - numpy_data_to_h5.py
    - train_pytorch.py
    - md_training_data
        - fcc
            - fcc_training.xyz
        - bcc 
            - bcc_training.xyz
        - hcp
            - hcp_training.xyz

The final sub folders containing the MD snapshots can also contain several files. Now you need to prepare the data for the fit. Therefore, you need to run:

```
  python extract_data.py [phase_name] [num_points]
```

Here, you need to replace the [phase_name] for example by 'fcc' and [num_points] by the number of points you want to use as input for the DG-CNN (8,16,32,64). This creates a folders named '1000000_[num_points]' containing .npy files with the training data. You need to repeat this procedure for every phase.
To prepare the dataset now for fitting we will pack now all the data from the .npy files into a .h5 file. 
This is done by:

```
  python numpy_data_to_h5.py 1000000_[num_points]
```

Finally, the model can be trained by:

```
  python train_pytorch.py [num_points]
```



# Disclaimer

This project and any code therein is not associated or affiliated to the OVITO GmbH. However, one of the authors (DU) is an employee of the OVITO GmbH.

# Acknowledgement

Work on this project was funded by the National High Performance Computing Center for Computational Engineering Science [NHR4CES](https://www.nhr4ces.de/).
