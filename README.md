# deinterf

The project aims to compensate for magnetic interference related to the magnetic field of the measurement platform itself within the magnetic total field data. Utilizing the Tolles-Lawson model and Ridge Regression, it models the relationship between flight attitude and magnetic interference, forming a magnetic compensation model. Herein, the flight attitude is obtained from magnetic tri-component measurements, while the magnetic interference is derived from the FOM flight measurement results after being processed through a high-pass filter.

## Getting started

`deinterf` can be installed using pip:

```shell
pip install git+https://github.com/dorian-li/deinterf.git
```
When updating, the following command can be used：
```
pip install --ignore-installed git+https://github.com/dorian-li/deinterf.git
```
A usage case can be found in the `example` folder

## Licensing

The code in this project is licensed under MIT license.
