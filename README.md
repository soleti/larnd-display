# Pixelated LArTPC event display

This package provides a web-based, interactive event display for pixelated LArTPCs. It is compatible with a variety of multi-modular detector geometries (Module-0, ProtoDUNE 2x2).

The application is deployed on [Spin](https://www.nersc.gov/systems/spin/) at [NERSC](https://www.nersc.gov), and is accessible from the web browser at [https://larnddisplay.lbl.gov/](https://larnddisplay.lbl.gov/).

## Installation

In case you would like to run the application locally, you can clone the repository and install it following these instructions.
The package requires [`larnd-sim`](https://github.com/DUNE/larnd-sim) but it doesn't need a GPU, so if you don't have one you can remove `cupy` from the list of packages needed by the [`setup.py`](https://github.com/DUNE/larnd-sim/blob/f0ffe09c62d3081cc38eead1f9b32f2e06d80667/setup.py#L16) of `larnd-sim`.

Once you have installed `larnd-sim` you can proceed with the installation of this package.

```bash
cd larnd-display
pip install .
```

## Usage

In order to run the event display you need to start the server:

```bash
evd.py /location/of/larnd-sim
```

where `/location/of/larnd-sim` is the directory where `larnd-sim` resides.

The event display will be available at the URL [http://localhost:5000/](http://localhost:5000/) and should like the screenshot below (for a Module0-like detector):

<img src='https://github.com/soleti/larnd-display/raw/main/assets/screenshot.png' width='100%'/>
