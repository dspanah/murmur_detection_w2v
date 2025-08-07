# murmur_detection_w2v

Python implementation of heart murmur detection using wav2vec 2.0 as presented in the following paper:

Davoud Shariat Panah, Andrew Hines, and Susan McKeever. "Exploring wav2vec 2.0 model for heart murmur detection." 2023 31st European Signal Processing Conference (EUSIPCO). IEEE, 2023.

## Setup
Install the required packages by running:

`pip install -r requirements.txt`

## Usage
To preprocess the Digiscope dataset use the `data preparation.ipynb` notebook under data driectory.
To fine tune and validate the model use `fine_tune.py` under src directory.
To test the model on test set use `test.py` under src directory.

## Citation

If you use this code, please cite the associated paper:

```bibtex
@inproceedings{panah2023exploring,
  title={Exploring wav2vec 2.0 model for heart murmur detection},
  author={Shariat Panah, Davoud and Hines, Andrew and McKeever, Susan},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)},
  pages={1010--1014},
  year={2023},
  organization={IEEE}
}

## Licence

This project is licensed under the Apache 2.0 License.
