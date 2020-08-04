from datetime import datetime
import argparse, logging
import dataset_annotation
import data_util
from dbn_wrapper import DbnWrapper
import config as cfg
from pathlib import Path
import time


scenarios = ['01', '02', '03', '04', '05', '06', '08']

parser = argparse.ArgumentParser()
parser.add_argument("data", help="Path to dataset")
parser.add_argument("dbn", help="Path to the xdsl (GeNIe) file that contains the (possibly untrained) dbn.")
parser.add_argument('--annotations', help='create all scene annotations', action='store_true')
parser.add_argument('--tvt', help='create training data for dbn for each scenario', action='store_true')
parser.add_argument("--scenario", choices=scenarios + ['all'], help="tells which scenario to proceed")
args = parser.parse_args()

# set variables in config.py
cfg.dbn_path = args.dbn
dbn = DbnWrapper(cfg.dbn_path)
cfg.dataset_root = args.data


# create annomations for all scenes in the dataset
if args.annotations:
    print("Start 'create_annotations'. Current Time:", datetime.now().strftime("%H:%M:%S"))
    max_distance = dataset_annotation.run_all(cfg.dataset_root)
    print("End 'create_annotations'. Current Time:", datetime.now().strftime("%H:%M:%S"))

# create training data file
if args.tvt:
    for scenario in scenarios:
        cfg.test_scenario = scenario

        # directory where the training data should be saved
        if cfg.train_set == 'small':
            directory = Path('training_data/small-slices' + str(dbn.get_slice_count())) / cfg.test_scenario
        elif cfg.train_set == 'total':
            directory = Path('training_data/slices' + str(dbn.get_slice_count())) / cfg.test_scenario
        else:
            assert False

        # if the directory already exists, I assume that the training file was already created
        if directory.exists():
            print(str(directory) + ' exists.')
            continue
        directory.mkdir(parents=True, exist_ok=True)

        # logger
        logger = logging.getLogger('data-creation')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(directory / 'data-creation.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        
        print("Start 'create_training_data'. Current Time:", datetime.now().strftime("%H:%M:%S"))
        dbn.create_training_data(directory)
        print("End 'create_training_data'. Current Time:", datetime.now().strftime("%H:%M:%S"))

# train and test scenarios
if args.scenario:
    if args.scenario != 'all':
        scenarios = [args.scenario]
    for scenario in scenarios:
        cfg.test_scenario = scenario
        
        # compute maximal distance between pedestrians in training data to discretize the variable (only variable that isn't discretized beforehand)
        max_distance = data_util.get_max_distance(cfg.dataset_root)
        dbn.set_distance_labels(max_distance)
        
        # directory that contains the training data and where the trained DBN will be saved
        if cfg.train_set == 'small':
            directory = Path('training_data/small-slices' + str(dbn.get_slice_count())) / cfg.test_scenario
        elif cfg.train_set == 'total':
            directory = Path('training_data/slices' + str(dbn.get_slice_count())) / cfg.test_scenario

        # create logger
        logger = logging.getLogger('tvt')
        logger.setLevel(logging.DEBUG)
        log_version = cfg.dbn_path[-8:-5]
        fh = logging.FileHandler(directory / ('tvt' + log_version + '.log'))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        # training
        print("Start 'dbn_training'. Current Time:", datetime.now().strftime("%H:%M:%S"))
        logger.info("Start 'dbn_training'. Current Time: " + str(datetime.now().strftime("%H:%M:%S")))
        dbn.train(directory / 'training_data.csv')
        logger.info("End 'dbn_training'. Current Time: " +  str(datetime.now().strftime("%H:%M:%S")))
        print("End 'dbn_training'. Current Time:", datetime.now().strftime("%H:%M:%S"))

        # inference
        print("Start 'dbn_testing'. Current Time:", datetime.now().strftime("%H:%M:%S"))
        dbn.testing()
        print("End 'dbn_testing'. Current Time:", datetime.now().strftime("%H:%M:%S"))


        
