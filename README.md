# Detection and Classification of Pneumonia in Chest X-ray Images
*Using a Multi-Network Pipeline Consisting of U-Net and ResNet*

Our end of semester project for the lecture "Advanced Machine Learning".

## Prerequisites

The code is developed for Python 3.7.
We use [Pipenv](https://pipenv.pypa.io/en/latest/) to manage dependencies.
With Pipenv available, just execute
```bash
pipenv sync  # installs the dependencies from Pipfile.lock
```
to install the dependencies required into a specific virtual environment.
(You might like to set `PIPENV_VENV_IN_PROJECT=1` before, which will place the
virtual environment within the repository.)

To start a shell within the environment, type `pipenv shell`.
Alternatively, prepend `pipenv run` to any desired command to be executed in the
environment.

## File structure

- [`datasets`](datasets): dataset classes
- [`network`](network): model and training related classes
- [`scripts`](scripts): executable scripts and CLIs for specific subtasks
- [`train`](train): CLIs to train U-Net and Full Network
- [`trafo`](trafo): transformation toolchain for data preprocessing and
                    augmentation
- [`utils`](utils): particular tools used by several other classes
