# OpenDS-CTS02

Each directory in the OpenDS-CTS02 folder represents a scene.
The first two digits of the directory's name represent the scenario to which the scene belong.

Each directory contains the following files:

* `p1*.txt`
* `p2*.txt`
* `log.txt`
* `pedestrian 1 features.csv`
* `pedestrian 2 features.csv`
* `dbn_annotations.csv`
* `dbn_annotations_continuous.csv`
* `pedestrian_positions.csv`
* `dbn_predictions.csv`

The files `p1*.txt` and `p2*.txt` are needed to create the OpenDS scene.
The suffix `_crossing` indicates which pedestrian is crossing the street (if any).
`p1` and `p2` denote the left and right pedestrian, respectively.


The file `log.txt` is the output of OpenDS, `pedestrian 1 features.csv`, `pedestrian 2 features.csv` are interpretations of `log.txt`.

The file `dbn_annotations.csv` contains the annotations for the EMIDAS-DBN and was obtained out of `pedestrian 1 features.csv` and `pedestrian 2 features.csv`.
While it contains the discretized values of the EMIDAS variables, `dbn_annotations_continuous.csv` contains the continuous values.

The file `pedestrian_positions.csv` contains the pedestrian positions throughout the scene.
The index column matches the index column of `dbn_annotations.csv` and `dbn_annotations_continuous.csv`.

The file `dbn_predictions.csv` contains the posterior probability distribution of the variables `p1_imas`, `p2_imas`, `p1_wti` and `p2_wti`.
These were computed using the EMIDAS-DBN that unrolls to 10 time slices.


# EMIDAS Source Code

The folder contains the following python files:

1) `emidas.py`
    Entry point to
    - generate all `dbn_annotations.csv` files for OpenDS-CTS02, or,
    - create the training data files where each of them leaves out a different test scenario, or,
    - train a DBN given the test scenario and run the predictions for the test scenario.

2) `config.py`
    File that serves as collection of global variables.
    If you use the code without calling `emidas.py` make sure that you set all global variables as needed.

3) `dbn_wrapper.py`
    Everything regarding the DBN: training, inference, explanations.

4) `dataset_annotation.py`
    Used to create the `dbn_annotations.csv` files, checks whether each scene is valid.

5) `data_util.py`
    Helper functions.

6) `util.py`
    Helper functions used in `dataset_annotation.py`.

# PySMILE

This project uses [SMILE](https://www.bayesfusion.com/smile/) ([C++ documentation](https://support.bayesfusion.com/docs/SMILE/), [Python wrapper documentation](https://support.bayesfusion.com/docs/Wrappers/), [download](https://download.bayesfusion.com/files.html?category=Academia)).

Warning: On macOS, Python should be installed with [official binaries](https://www.python.org/downloads/mac-osx/). PySMILE is not compatible with binaries installed from other sources (e.g. Homebrew repo).

This folder contains the following PySMILE libraries:

- `pysmile.so` PySMILE 1.5 for macOS using Python 3.8
- `pysmile.pyd` PySMILE 1.5 for Windows x64 using Python 3.7
- `pysmile_license.py` issued for me (Nora) until 31.12.2020 (see the [BayesFusion download site](https://download.bayesfusion.com/files.html?category=Academia) for information on the licensing)


# Source Code Snippets

**Example how to train a DBN:**

```python
# Training
cfg.dbn_path = 'emidas-dbn.xdsl' # save path of DBN in config.py
dbn = DbnWrapper(Path(cfg.dbn_path)) # create DbnWrapper object
dbn.train(Path('train_data.csv')) # train the DBN
```

It is mandatory that each column of `train_data.csv` matches a DBN variable and that the values in `train_data.csv` match the values in the DBN.
The trained DBN is saved at the location of the training data with `_trained` suffix.

**Example how to compute the predictions of some scene:**

```python
# Prediction
cfg.dbn_path = 'emidas-dbn.xdsl' # save path of DBN in config.py
dbn = DbnWrapper(Path(cfg.dbn_path), trained=True) # create DbnWrapper object
dbn.prediction(feature_file='dbn_annotations.csv', save_to='dbn_prediction.csv') # run prediction
```

Each line in `dbn_prediction.csv` contains the IMAS/WTI prediction for the next time step.

**Example to compute the most relevant explanations:**

In the following example, we want the most relevant explanations for the prediction of `p1_imas = very_high` in the next time step (`time_limit=1`), using the feature subset `[p1_head, p1_body]`.

```python
# MRE
dbn = DbnWrapper('emidas-dbn.xdsl', trained=True, bayesian_algorithm=3)
dbn.most_relevant_explanation('mre.csv', {'p1_imas' : 'very_high'}, ['p1_head', 'p1_body'], time_limit=1)
```


**Example to create evidence balance sheet:**

Creates an evidence balance sheet for the results in line `107` of `dbn_prediction.csv`.

```python
# Evidence balance sheet
dbn = DbnWrapper('emidas-dbn.xdsl', trained=True)
df = pd.read_csv('dbn_prediction.csv', usecols=['p1_head', 'p2_head', 'p1_body', 'p2_body', 'p1_approach', 'p2_approach', 'p1_gesture', 'p2_gesture', 'dist'])
row = 107 # row of interest
row_start = max(0, row - dbn.get_slice_count() + 2)
dbn.evidence_sheet('p1_imas', 'very_high', df.iloc[row_start:row+1], 'evidence_sheet.csv')
```

Use `groups=True` in `DbnWrapper.evidence_sheet` to get an evidence balance sheet where the features are not partitioned according to their value.