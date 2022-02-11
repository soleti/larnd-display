# LArND event display

This package provides a web-based event display for LArPix data. It is compatible with a variety of multi-modular detector geometries (Module-0, ProtoDUNE 2x2, ND-LAr, etc.).

## Installation

```bash
python setup.py install
```

## Usage

```bash
evd.py input_file.h5 detector_properties.yaml pixel_layout.yaml
```

where `input_file.h5` is a HDF5 file containing the event data in the LArPix format, `detector_properties.yaml` is a YAML file containing the detector properties (as the ones [here](https://github.com/DUNE/larnd-sim/tree/master/larndsim/detector_properties)), and `pixel_layout.yaml` is a YAML file containing the pixel layout (as the ones in [here](https://github.com/DUNE/larnd-sim/tree/master/larndsim/pixel_layouts)).