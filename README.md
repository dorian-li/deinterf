# deinterf

The project aims to compensate for magnetic interference related to the magnetic field of the measurement platform itself within the magnetic total field data. Utilizing the Tolles-Lawson model and Ridge Regression, it models the relationship between flight attitude and magnetic interference, forming a magnetic compensation model. Herein, the flight attitude is obtained from magnetic tri-component measurements, while the magnetic interference is derived from the FOM flight measurement results after being processed through a high-pass filter.

## Getting started

deinterf can be installed using pip:

```shell
pip install deinterf
```

A usage case can be found in the `example` folder

## Manual installation

```shell
git clone https://github.com/dorian-li/deinterf.git
cd deinterf
pip install -e .
# If editable mode is not required:
# pip install .
# This means that modifications made within the cloned code cannot be directly fed back to the running code and need to be reinstalled.
```

## Licensing

The code in this project is licensed under MIT license.