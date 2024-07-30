# Torch Lab

## Overview

Torch Lab is a hackable template for artificial intelligence and machine learning research projects using the 
PyTorch Ecosystem.

## Project Structure

`torchlab.data/` contains code for the Hugging Face Dataset and Dataloaders.

### Project Root

`checkpoints` directory contains training checkpoints and the pre-trained production model.

`data` directory for local data caches.

`docs` directory for technical documentation.

`logs` directory contains logs generated from experiment managers and profilers.

`notebooks` directory can be used to present EDA and demo notebooks.

`requirements` directory of requirement files titled by purpose.

`tests` module contains unit and integration tests targeted by pytest.

`setup.py` `setup.cfg` `pyproject.toml` and `MANIFEST.ini` assist with packaging the Python project.

`.pre-commit-config.yaml` is required by pre-commit to install its git-hooks.

## Installation

The recommended installation is as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
