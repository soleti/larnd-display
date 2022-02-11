#!/usr/bin/env python

import setuptools

VER = "0.1.0"

setuptools.setup(
    name="larnd-display",
    version=VER,
    author="DUNE collaboration",
    author_email="roberto@lbl.gov",
    description="Event display for modular pixelated LArTPCs",
    url="https://github.com/soleti/larnd-display",
    packages=setuptools.find_packages(),
    scripts=["evd.py"],
    install_requires=["h5py", "dash", "numpy", "plotly", "fire", "pyyaml", "particle"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: by End-User Class :: Developers",
        "Operating System :: Grouping and Descriptive Categories :: OS Independent (Written in an interpreted language)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.7',
)
