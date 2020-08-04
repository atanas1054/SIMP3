Directory content:

- `pyemidas` contains all the source code related to EMIDAS. See `pyemidas/README.md` for more detailed information

- `dataset` contains files and python script that were part of the creation process of the dataset OpenDS-CTS02. The source code that creates the scenes is *not* included in there.

- `genie-models` contains untrained DBN in the subdirectory `untrained`. The subdirectory `trained` contains all trained DBN mentioned in the thesis, that were trained on the full OpenDS-CTS02 dataset. The subdirectory `trained-small` contains all trained DBN mentioned in the thesis, that were trained on the randomly selected subset of the OpenDS-CTS02 dataset.

- `study` contains the results of the study and some plots for visualization

- `OpenDS-CTS02-compressed` contains the total dataset (zipped in groups to not exceed the 100MB file size)
