<!-- -*- gfm -*- -->

# CS7NS1 Group 14 Project 2: CAPTCHA Classification

### Running classification on a Raspberry Pi

```sh
./run.sh
```

This will take care of creating and activating a Python virtual environment
under a directory called `.venv` if needed, and will install the necessary
dependencies, namely `Pillow` and `tflite_runtime`.  It will then proceed to
classify the 1000 images under `img/contovob` and write the results to
`results/classified.csv` before deactivating the virtual environment.  You can
safely invoke `run.sh` repeatedly; it will try to detect when a virtual
environment is present or active, and act accordingly.

To change these defaults, see the supported options described under `./run.sh -h`.

**Note:** The `run.sh` script assumes your Python version is 3.8 and your
machine's architecture is `armv7l`.  If they are something else, you must change
the corresponding line in `requirements.txt` accordingly.  You can query your
Python version with `python3 --version` and your machine's architecture with
`uname -m`.  For architectures supported by `tflite_runtime`, see
https://www.tensorflow.org/lite/guide/python.

**Note:** You may see Pip errors about failing to build Pillow, but you can
ignore these as the installation should still succeed.

### Files

* `generate.py`: Generates labelled CAPTCHA images for training.  The labels are
  encoded as URL-safe Base64 to allow for special characters not supported by
  the system.  This file was rewritten based on the eponymous one provided by
  the lecturer.

* `train.py`: Trains a CNN on given CAPTCHAs.  This is a lightly modified
  version of the eponymous file provided by the lecturer.  The modifications
  include:
  - Error handling for when the current batch of files becomes empty.
  - URL-safe Base64 decoding of image labels.
  - Removal of manual TensorFlow device placement.
  - Reduction of learning rate and `EarlyStopping` patience.
  - Addition of CSV logging.
  - Some simplification and prettification of the code.

* `classify.py`: Almost the same file provided by the lecturer for running full
  TensorFlow inference using a H5 model.  The only modifications are sorted
  output and stripping of space characters from the classifications.

* `classifylite.py`: A rewritten version of `classify.py` for running on a
  Raspberry Pi, which replaces the OpenCV library with Pillow, and TensorFlow
  with the TensorFlow Lite runtime interpreter.

* `convertf.py`: A helper script for converting H5 models to TFLite models.  It
  is made redundant by the `tflite_convert` script that is often bundled with
  TensorFlow.

* `merge-csv.py`: A helper script for merging CSV files containing output
  classifications.  This was used for correcting some of the model's
  classifications with classifications that were manually verified against
  Submitty.

* `run.sh`: Automation script wrapping `classifylite.py` in a Python virtual
  environment.

* `requirements.txt`: List of Python package dependencies understood by Pip.

* `img/`: Contains subdirectories with the per-student assigned CAPTCHA image
  sets.

* `models/`: Contains the four trained models.  All of our submissions to
  Submitty were classified by `model1.tflite`.  Each model also has some log
  files associated with it, which contain TensorFlow statistics collected during
  the model's training.

* `results/`: Contains CSV files containing various classification results in a
  format intended for Submitty.

* `syms/`: Contains various symbol set files.  All of our submissions to
  Submitty were classified with the `p2.txt` symbol set.

### Metrics

Training and conversion from H5 to TFLite was carried out on an `x86_64` Linux
machine in the LG-12 lab with 1 GPU, 12 CPU cores, and 16 GB RAM.

Classification was carried out on an `armv7l` Raspberry Pi with 4 CPU cores and
4 GB RAM.

| Metric                                    | Value              |
| ----------------------------------------- | ------------------ |
| Number of training images                 | 160,000            |
| Number of validation images               | 16,000             |
| Total size of training set                | 1.5 GiB            |
| Number of images classified               | 1000               |
|                                           |                    |
| Time taken by `generate.py`               | 10-30 mins         |
| Time taken by `train.py` with GPU         | 30-60 mins / epoch |
| Number of epochs                          | 12-16              |
| Time taken by `classifylite.py`           | 1-2 mins           |
| Time taken by `run.sh`                    | <5 mins            |
|                                           |                    |
| Size of H5 model                          | 46 MiB             |
| Size of TFLite model                      | 12 MiB             |
| Total model parameters                    | 2,973,004          |
| Trainable model parameters                | 2,970,060          |
| Non-trainable model parameters            | 2,944              |
| Number of model inputs                    | 1                  |
| Number of model outputs                   | 6                  |
| Number of `Conv2D` layers                 | 10                 |
| Batch size                                | 32                 |
| Adam optimiser learning rate              | `1e-4`             |
|                                           |                    |
| Accuracy of model on Submitty             | 34-35%             |
| Accuracy drop from H5 → TFLite conversion | None               |
| Training loss in last epoch               | 0.0554             |
| Training accuracy for each character      | 99%                |
| Validation loss in last epoch             | 1.0083             |
| Validation accuracy for each character    | 90-98%             |
| Accuracy gain from smaller symbol set     | None               |
| Preprocessing                             | None               |
