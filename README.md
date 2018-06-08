# Seigen: Seismology through code generation

## Overview

[Seigen](http://www.opesci.org) is an elastic wave
equation solver for seimological problems based on the
[Firedrake](http://www.firedrakeproject.org) finite element
framework. It forms part of the [OPESCI](http://www.opesci.org)
seismic imaging project.

## Quickstart

Seigen requires the installation of Firedrake and must be run from
within the Firedrake virtual environment (venv). To first install Firedrake
please follow these [instructions](http://www.firedrakeproject.org/download.html#).

Once Firedrake is installed and the venv is activated, you can install
Seigen using the following commands:

```
git clone https://github.com/opesci/seigen.git
pip install -e seigen
```

## Documentation

The documentation can be built by navigating to the `docs` directory and running `make html` at the command line. A new `build` folder will then be created containing the documentation in HTML format.

Details regarding the model equations and the discretisation and solution procedure are provided, along with a description of the pulse propagation application (see `tests/pulse`). This acts as an example of how to set up a simulation in Seigen.

## Licence

Seigen is open-source software and is released under the [MIT License](https://opensource.org/licenses/MIT). See the file `LICENSE.md` for more information.

## Contact

Comments and feedback may be sent to opesci@imperial.ac.uk.
