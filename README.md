# CalRatioTrainer

Train the CalRatio 2019 RNN

## Usage

## Installation

Installation instructions are generally tricky: this really needs to be trained on a GPU.

### WSL2

This is without using the GPU (so good for testing).

1. Open up an instance. If you are using the full Run 2 dataset, then you'll need the 40GB instance to be as efficient as possible.
1. Open a terminal window
1. `git clone https://github.com/gordonwatts/CalRatioTrainer.git` into whatever directory you want to run out of, in a new virtual environment.
1. `cd CalRatioTrainer`
1. `pip install -e .[wsl2]`
    * If you want to do development, etc., then do `pip install -e .[test,wsl2]`

This should work anywhere you are using a clean environment. It will install `TensorFlow`, for example. It is always a fight getting the right version of TF and the underlying GPU libraries to work together, so you may have to fiddle after the install depending on your setup. Feel free to submit PR's if you find something that might be interesting to others!

### Chicago Analysis Facility

The installation is expected to take place on a Jupyter instance where the proper TF libraries have already been installed.

1. Open up an instance. If you are using the full Run 2 dataset, then you'll need the 40GB instance to be as efficient as possible.
1. Open a terminal window
1. `git clone https://github.com/gordonwatts/CalRatioTrainer.git` into whatever directory you want to run out of.
1. `cd CalRatioTrainer`
1. `pip install -e .`
    * If you want to do development, etc., then do `pip install -e .[test]`

You should be ready to go!

## Acknowledgements

This is based on the work originally done by Felix in the CalRatio group in ATLAS. This RNN was published in xxx.
The running and design has been improved since then.
