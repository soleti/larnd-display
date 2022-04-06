# Pixelated LArTPC event display

This package provides a web-based, interactive event display for pixelated LArTPCs. It is compatible with a variety of multi-modular detector geometries (Module-0, ProtoDUNE 2x2, ND-LAr, etc.).

## Installation

The package requires [larnd-sim](https://github.com/DUNE/larnd-sim) but it doesn't need a GPU, so if you don't have one you can remove `cupy` from the list of packages needed by the `setup.py` of `larnd-sim`.

Once you have installed `larnd-sim` you can proceed with the installation of this package.

```bash
cd larnd-display
pip install .
```

## Usage

In order to run the event display you need to start the server:

```bash
evd.py detector_properties.yaml pixel_layout.yaml
```

where `detector_properties.yaml` is a YAML file containing the detector properties (as the ones [here](https://github.com/DUNE/larnd-sim/tree/master/larndsim/detector_properties)), and `pixel_layout.yaml` is a YAML file containing the pixel layout (as the ones in [here](https://github.com/DUNE/larnd-sim/tree/master/larndsim/pixel_layouts)).

The event display will be available at the URL `http://localhost:8000/` and should like the screenshot below (for a Module0-like detector):

<img src='https://github.com/soleti/larnd-display/raw/main/docs/screenshot.png' width='100%'/>
