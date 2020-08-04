import pysmile
import pysmile_license
from numpy import log2
from pathlib import Path
import csv, glob, re, logging, itertools, os
import pandas as pd
from numpy import log10, log2

import config as cfg
import data_util


class DbnWrapper:
    """Object implementing als necessary functions for the DBN.
    """

    def __init__(self, dbn_file, bayesian_algorithm=0, trained=False):
        """Constructor

        Args:
            dbn_file (pathlib.Path): absolute path to the xdsl network file
            bayesian_algorithm (int, optional): Inference algorithm that should be used. Defaults to 0.
            trained (bool, optional): Takes the distance discretization from the DBN definition if it is already trained. Defaults to False.
        """
        self.net = pysmile.Network()
        self.net.read_file(str(dbn_file))
        self.trained = False
        self.outcome_str_to_num = {'very_low' : '1', 'low' : '2', 'medium' : '3', 'high' : '4', 'very_high' : '5'}
        self.outcome_num_to_str = {1 : 'very_low', 2 : 'low', 3 : 'medium', 4 : 'high', 5 : 'very_high'}

        if bayesian_algorithm != 0:
            self.net.set_bayesian_algorithm(bayesian_algorithm)

        if trained:
            self.dist_labels = self.net.get_outcome_ids('dist')
            last_bin = int(self.dist_labels[-1].split('_')[1])
            self.dist_bins = [d for d in range(0, last_bin+1, 3)] + [float("inf")]



    def get_slice_count(self):
        """Returns the number of slices of the DBN.

        Returns:
            int: number of slices
        """
        return self.net.get_slice_count()

    
    def get_all_node_ids(self):
        """Returns the identifier of all variables in the DBN.

        Returns:
            list: names of each variable
        """
        return self.net.get_all_node_ids()

    
    def get_features_ids(self):
        """Returns the identifier of the feature nodes in the DBN.

        Returns:
            list: names of each feature variable
        """
        if not hasattr(self, 'feature_ids'):
            # Get all feature nodes (= nodes without normal (non-temporal) arcs)
            self.features_ids = []
            for h in self.net.get_all_nodes():
                children = self.net.get_children(h)
                if len(children) == 0:
                    node_id = self.net.get_node_id(h)
                    self.features_ids.append(node_id)
        return self.features_ids


    def set_distance_labels(self, max_distance):
        """Given the maximal distance, creates the discretized labels.

        Args:
            max_distance (int): maximal distance in the training data
        """
        step = 3
        bins = [d for d in range(0, max_distance, step)] + [float("inf")]
        labels = []
        for i in range(1, len(bins)):
            labels.append('range_' + str(bins[i-1]) + '_' + str(bins[i]))

        self.dist_bins = bins
        self.dist_labels = labels

        dist_node = self.net.get_node('dist')
        count = self.net.get_outcome_count(dist_node)
        cpt_num_cols = int(len(self.net.get_node_definition(dist_node)) / count)
        if len(labels) < count:
            val = 1/len(labels)
            node_cpt = ([val] * len(labels) + [0] * (count-len(labels))) * cpt_num_cols
            self.net.set_node_definition(dist_node, node_cpt)

        index = 0
        for i in range(min(count, len(labels))):
            self.net.set_outcome_id(dist_node, i, labels[i])
            index += 1
        
        for j in range(index, len(labels)):
            self.net.add_outcome(dist_node, labels[j])
            index += 1

        for _ in range(index, count):
            self.net.delete_outcome(dist_node, index)

    
    def target_nodes(self, unrolled=False):
        """returns the target nodes. If the DBN was unrolled to be able to run the validation or test functions, this function has to return the target variables for each time step.

        Args:
            unrolled (bool, optional): If the DBN was unrolled. Defaults to False.

        Returns:
            list: list of target variables
        """
        nodes = ['p1_imas', 'p2_imas', 'p1_wti', 'p2_wti']
        if not unrolled:
            return nodes
        ret_list = nodes + [node + '_' + str(i) for i in range(1, 10) for node in nodes]
        return ret_list


    
    def create_smile_data_file(self, file_path, is_target_file):
        """Creates a data file which matches the file requirements of GeNIe/SMILE.

        Args:
            file_path (str): path to which the file should be saved
            is_target_file (str -> bool): function that check if file is target file (train or test)

        Returns:
            int: number of lines in the created file
        """
        num_lines = 0

        dbn_time_steps = self.get_slice_count()
        frame_shift = 1

        # names of dbn nodes
        node_names = self.get_all_node_ids()

        with open(file_path, 'w+') as output:
            fieldnames_output = node_names + [name + '_' + str(i) for i in range(1, dbn_time_steps) for name in node_names]
            writer = csv.DictWriter(output, fieldnames=fieldnames_output)
            writer.writeheader()

        # go through all annotation files
        for subdir, _, files in os.walk(cfg.dataset_root):
            for f in files:
                path_to_file = Path(subdir) / f
                if is_target_file(path_to_file):
                    df = pd.read_csv(path_to_file, index_col=0)
                    df = df.rename(data_util.renaming_cols, axis='columns')
                    df['dist'] = pd.cut(df['dist'], bins=self.dist_bins, labels=self.dist_labels, right=False) # discretize distance
                    df = df.replace({'p1_imas' : self.outcome_num_to_str, 'p2_imas' : self.outcome_num_to_str, 'p1_wti' : self.outcome_num_to_str, 'p2_wti' : self.outcome_num_to_str}) # replace int representation (1/2/3/4/5) into string representation (very_low, low, ...)
                    df = df[node_names] # new column order
                    for col in node_names:
                        df[col] = df[col].astype('category')

                    # Create csv matching the required data format for DBN parameter learning
                    train_df = pd.DataFrame()
                    for i in range(dbn_time_steps):
                        subset = (df[i:])[::frame_shift]                    # each (frame_shift)-th row starting from the i-th row
                        subset = subset.rename(columns=data_util.append_index(i))     # renamed column names
                        subset = subset.reset_index(drop=True)              # reset index numbers
                        train_df = pd.concat([train_df, subset], axis=1)
                    
                    train_df = train_df.dropna()                            # drop each line which is "shorter" than needed

                    with open(file_path, 'a') as output:
                        train_df.to_csv(output, header=False, index=False, line_terminator='\n')
                    
                    num_lines += len(train_df.index)

        return num_lines


    def create_training_data(self, directory):
        """Creates training data file. First, the function checks the highest value of the variable distance in the training data, because the discretization is done during the file creation.

        Args:
            directory (pathlib.Path): directory where the file should be saved to
        """
        logger = logging.getLogger('data-creation')

        max_distance = data_util.get_max_distance(cfg.dataset_root)
        logger.info('The maximal distance is ' + str(max_distance))
        self.set_distance_labels(max_distance)

        num_lines_train = self.create_smile_data_file(directory / 'training_data.csv', data_util.is_train_annotations_file)
        logger.info('The training data file has ' + str(num_lines_train) + ' lines.')


 
    def train(self, train_data_path):
        """Train the DBN and update network in file.

        Parameters:
            train_data_path (str): path of training data file

        """
        # the training data 
        train_set = pysmile.learning.DataSet()
        train_set.read_file(str(train_data_path))

        # learning
        matching = train_set.match_network(self.net)
        em = pysmile.learning.EM()
        em.set_randomize_parameters(True)
        em.learn(train_set, self.net, matching)

        # save parameters to file
        path = train_data_path.parent / (Path(cfg.dbn_path).stem + '_trained.xdsl')
        self.net.write_file(str(path))

        print('DBN was trained on dataset', train_data_path)


    def validation(self, validation_data_path, folds):
        """WARNING: validation with Genie/SMILE does not work on DBNs yet. To use it, it is necessary to unroll the DBN (I wouldn't recomend to do this, other functions won't work on the unrolled DBN). This function isn't used in the project, it might not work correctly.

        Args:
            validation_data_path (pathlib.Path): absolute path of data file
            folds (int): number of folds
        """
        logger = logging.getLogger('tvt')

        # the validation data 
        val_set = pysmile.learning.DataSet()
        val_set.read_file(str(validation_data_path))

        # unrolled network
        self.unrolled_net = self.net.unroll().unrolled

        # validation
        matching = val_set.match_network(self.unrolled_net)
        validator = pysmile.learning.Validator(self.unrolled_net, val_set, matching)
        for node_handle in self.target_nodes(unrolled=True):
            validator.add_class_node(node_handle)
        em = pysmile.learning.EM()
        em.set_randomize_parameters(True)
        validator.k_fold(em, folds)
        for node_handle in self.target_nodes(unrolled=True):
            for outcome in range(self.unrolled_net.get_outcome_count(node_handle)):
                logger.info('Accurracy of ' + node_handle + ' ' + self.unrolled_net.get_outcome_id(node_handle, outcome) + ': ' + str(validator.get_accuracy(node_handle, outcome)))

        # save validation parameters to file
        path = validation_data_path.parent / (Path(cfg.dbn_path).stem + '_validated.xdsl')
        self.unrolled_net.write_file(str(path))
        
        # result dataset
        result_set = pysmile.learning.DataSet()
        result_set = validator.get_result_data_set()
        file_name = validation_data_path.parent / ('result-validation' + cfg.dbn_path[-8:-5] + '.csv')
        i = 0
        while file_name.exists():
            i += 1
            file_name = validation_data_path.parent / ('result-validation' + cfg.dbn_path[-8:-5] + '-' + str(i) + '.csv')
        result_set.write_file(str(file_name), ',', '', True)



    def test(self, test_data_path, unrolled=False):
        """WARNING: validation with Genie/SMILE does not work on DBNs yet. To use it, it is necessary to unroll the DBN (I wouldn't recomend to do this, other functions won't work on the unrolled DBN). This function isn't used in the project, it might not work correctly. "Testing" is perfomed using the self.testing function.

        Args:
            test_data_path (pathlib.Path): absolute path of data file
            unrolled (bool, optional): Whether the DBN is unrolled. Defaults to False.
        """
        logger = logging.getLogger('tvt')

        # the test data 
        test_set = pysmile.learning.DataSet()
        test_set.read_file(str(test_data_path))

        if unrolled:
            net = self.unrolled_net
        else:
            net = self.net

        # testing
        matching = test_set.match_network(net)
        validator = pysmile.learning.Validator(net, test_set, matching)
        for node_handle in self.target_nodes(unrolled):
            validator.add_class_node(node_handle)
        validator.test()

        # result dataset
        result_set = pysmile.learning.DataSet()
        result_set = validator.get_result_data_set()
        file_name = test_data_path.parent / ('result-test' + cfg.dbn_path[-8:-5] + '.csv')
        i = 0
        while file_name.exists():
            i += 1
            file_name = test_data_path.parent / ('result-test' + cfg.dbn_path[-8:-5] + '-' + str(i) + '.csv')
        result_set.write_file(str(file_name), ',', '', True)

        # Confusion Matrices
        for node in self.target_nodes(unrolled):
            logger.info('Confusion matrix of ' + node + ': ' + str(validator.get_confusion_matrix(node)))


    def update_and_return_target_belief(self, evidence_data, target_node_ids, clear_evidence=True, undo=False, virtual_evidence=None):
        """Inserts all given data in the DBN, updates the posterior probability of the unobserved (target) nodes and returns their probability at the last time step.

        Args:
            evidence_data (pd.DataFrame): data that should be inserted into the DBN
            target_node_ids (list): target nodes
            clear_evidence (bool, optional): Whether previous evidence should be deleted before inserting the new data. Defaults to True.
            undo (bool, optional): Whether the newly inserted data (not all data) should be removed at the end of the function. Defaults to False.
            virtual_evidence (dict, optional): Virtual evidence for some nodes at slice 0. Defaults to None.

        Raises:
            ValueError: if the evidence_data parameter hasn't the right shape

        Returns:
            dict: Posterior probabilities of each target variable for each value
        """
        if clear_evidence:
            self.net.clear_all_evidence()
            self.net.clear_all_targets()

        evidence_data = evidence_data.reset_index(drop=True)

        evidence_data_cols = set(evidence_data.columns)
        if evidence_data_cols == set(['feature', 'time', 'value']):
            for row in range(len(evidence_data.index)):
                self.net.set_temporal_evidence(node_id=evidence_data.loc[row, 'feature'],
                                               slice=evidence_data.loc[row, 'time'], 
                                               outcome_id=evidence_data.loc[row, 'value'])
            target_slice = self.net.get_slice_count() - 1

        elif evidence_data_cols == set([feature for feature in self.get_features_ids() if not feature[-1].isdigit()]):
            for row in range(len(evidence_data.index)):
                for feature in evidence_data_cols:
                    self.net.set_temporal_evidence(node_id=feature, slice=row, outcome_id=evidence_data.loc[row, feature])
            target_slice = len(evidence_data.index)

            if virtual_evidence:
                assert self.net.get_outcome_ids('p1_imas') == ['very_high', 'high', 'medium', 'low', 'very_low']
                for key, evidence_list in virtual_evidence.items():
                    self.net.set_virtual_evidence(node_id=key, evidence=evidence_list)
            
        else:
            raise ValueError('Parameter evidence_data should be a DataFrame with three columns named feature, time and value or a colums for each feature in the DBN.')

        # this reduces the workload since all variables are targets when no target is set
        for target in target_node_ids:
            if not self.net.is_target(target):
                self.net.set_target(target, True)
        
        # computes the posterior probability distributen of all unobserved target variables (using the inference algorithm set in the constructor)
        self.net.update_beliefs()

        # get P(target | new evidences, old evidences) (there might be no old evidence)
        return_dict = {}
        for target in target_node_ids:
            target_handle = self.net.get_node(target)
            outcome_count = self.net.get_outcome_count(target_handle)
            node_probs = self.net.get_node_value(target_handle)
            node_probs = node_probs[target_slice * outcome_count : (target_slice + 1) * outcome_count]                                
            return_dict[target] = {self.net.get_outcome_id(target_handle, outcome) : node_probs[outcome] for outcome in range(outcome_count)}

        # remove added evidence
        if undo:
            if evidence_data_cols == set(['feature', 'time', 'value']):
                for row in range(len(evidence_data.index)):
                    self.net.clear_temporal_evidence(node_id=evidence_data.loc[row, 'feature'],
                                                slice=evidence_data.loc[row, 'time'])

            elif evidence_data_cols == set(self.get_features_ids()):
                for row in range(len(evidence_data.index)):
                    for feature in evidence_data_cols:
                        self.net.clear_temporal_evidence(node_id=feature, slice=row)

            self.net.update_beliefs()

        return return_dict


    def prediction(self, feature_file, save_to):
        """Writes the posterior probability distribution into annotations files (file is saved separately).

        Args:
            feature_file (str): path to CSV file that contains the DBN annotations
            save_to (str): path to which the prediction file should be saved to 
        """
        df_features = pd.read_csv(feature_file, index_col=0)
        df_features = df_features.rename(data_util.renaming_cols, axis='columns') # rename columns according to DBN scheme
        
        # discretize distance according to dbn labels
        df_features['dist'] = pd.cut(df_features['dist'], bins=self.dist_bins, labels=self.dist_labels, right=False)

        # dict to save predictions
        prediction = {'p1_imas' : {'very_high' : [], 'high' : [], 'medium' : [], 'low' : [], 'very_low' : []},
                    'p2_imas' : {'very_high' : [], 'high' : [], 'medium' : [], 'low' : [], 'very_low' : []},
                    'p1_wti' : {'very_high' : [], 'high' : [], 'medium' : [], 'low' : [], 'very_low' : []},
                    'p2_wti' : {'very_high' : [], 'high' : [], 'medium' : [], 'low' : [], 'very_low' : []}}

        dbn_time_slices = self.net.get_slice_count()

        for row in range(len(df_features.index)):
            row_start = max(0, row - dbn_time_slices + 2)
            cols_idx = [df_features.columns.get_loc(col) for col in self.get_features_ids() if not col[-1].isdigit()]
            try:
                probs = self.update_and_return_target_belief(df_features.iloc[row_start:row+1, cols_idx], prediction.keys())
            except Exception as e:
                logger = logging.getLogger('tvt')
                logger.error('Error during prediction of file ' + str(feature_file) + ': ' + str(e))
                print('Error during prediction of file ' + str(feature_file) + '. See tvt.log for the error message.')
                return
            
            assert set(prediction.keys()) == set(probs.keys())

            # save precitions into dict
            for target in probs:
                for outcome in probs[target]:
                    prediction[target][outcome].append(probs[target][outcome])

        # save precitions in data frame
        for node in prediction:
            for outcome in prediction[node]:
                col_name = node + '_' + self.outcome_str_to_num[outcome] + '_pred'
                df_features[col_name] = prediction[node][outcome]
                df_features[col_name] = df_features[col_name].apply(lambda cell: round(cell,  6))

        # custom columns order
        new_cols = list(filter(lambda col : '_pred' in col, df_features.columns))
        col_order = self.get_all_node_ids()
        for target in ['p1_imas', 'p2_imas', 'p1_wti', 'p2_wti']:
            i = col_order.index(target)
            col_order = col_order[:i+1] + list(filter(lambda col : target in col, new_cols)) + col_order[i+1:]
        df_features = df_features[col_order]

        # save prediction file
        df_features.to_csv(save_to)


    def testing(self):
        """Creates all predictions given the configuration in config.py.
        """
        if cfg.location_pred == 'benchmark':
            prediction_root = cfg.dataset_root
        else:
            if cfg.train_set == 'small':
                prediction_root = Path('../predictions') / (cfg.test_scenario + '-small') / str(Path(cfg.dbn_path).stem).replace('_trained', '')
            else:
                prediction_root = Path('../predictions') / cfg.test_scenario / str(Path(cfg.dbn_path).stem).replace('_trained', '')
            prediction_root.mkdir(parents=True, exist_ok=True)

        for root, _, files in os.walk(cfg.dataset_root):
            root = Path(root)
            for f in files:
                path_to_f = root / f
                if data_util.is_test_annotations_file(path_to_f):
                    if cfg.location_pred == 'benchmark':
                        save_to = str(path_to_f).replace('_annotations', '_prediction')
                    else:
                        save_to = prediction_root / root.name / 'dbn_prediction.csv'
                        save_to.parent.mkdir(exist_ok=True)
                    if not save_to.exists(): # to avoid that a scene is predicted again
                        self.prediction(path_to_f, save_to)
        
        data_util.confusion_matrix(prediction_root)



    def evidence_sheet(self, target_node_id, target_outcome_id, evidence_data, output_file, groups=True, method='bayes'):
        """Create the evidence balance sheet on the basis of the given data.

        Args:
            target_node_id (str): the target node of the evidence sheet
            target_outcome_id (str): the target outcome of the evidence sheet
            evidence_data (pd.DataFrame): the dataframe containg the data for which the evidence sheet is created
            output_file (str): path to which the output should be saved
            groups (bool, optional): whether or not the features should be grouped by outcome, defaults to True.
            method (str, optional): whether to use the weight of evidence ('bayes') or the cross-entropy ('entropy'), defaults to 'bayes'.

        Raises:
            ValueError: if the 'method' argument is not correct
        """
        method_types = ['bayes', 'entropy']
        if method not in method_types:
            raise ValueError(' Invalid method type. Expected one of: %s' % method_types)

        evidence_data = evidence_data.reset_index(drop=True)

        # building groups of temporal variables
        evidence_groups = []
        if groups:
            # group by feature and value
            for feature in evidence_data.columns:
                curr_group = {}
                for row in range(len(evidence_data.index)):
                    if not curr_group:
                        curr_group = {'feature' : [feature], 'value' : [evidence_data.loc[row, feature]], 'time' : [row]}
                    elif curr_group['value'][0] == evidence_data.loc[row, feature]:
                        curr_group['time'].append(row)
                        curr_group['feature'].append(feature)
                        curr_group['value'].append(evidence_data.loc[row, feature])
                    elif curr_group['value'][0] != evidence_data.loc[row, feature]:
                        df = pd.DataFrame(curr_group)
                        evidence_groups.append(df)
                        curr_group = {'feature' : [feature], 'value' : [evidence_data.loc[row, feature]], 'time' : [row]}
                    else:
                        assert False
                df = pd.DataFrame(curr_group)
                evidence_groups.append(df)
        else:
            # group by feature only
            for feature in evidence_data.columns:
                df = pd.DataFrame({'feature' : [feature] * len(evidence_data.index), 'value' : evidence_data[feature],  'time' : list(evidence_data.index)})
                evidence_groups.append(df)

        with open(output_file, 'w') as output:
            if method == 'bayes':
                fieldnames = ['Feature', 'Value', 'Slices', 'WOE', 'Target Probability']
            else:
                fieldnames = ['Indicant', 'State', 'Slices', 'Entropy', 'Target Probability']
            writer = csv.writer(output)
            writer.writerow(fieldnames)

        df_evidence = pd.DataFrame(columns=['feature', 'value', 'time'])
        while evidence_groups:
            max_val = - float('inf')
            max_val_group_idx = None
            prob_sheet = None

            last_probs = self.update_and_return_target_belief(df_evidence, [target_node_id], clear_evidence=True, undo=False)
            last_prob = last_probs[target_node_id][target_outcome_id]

            if df_evidence.empty:
                with open(output_file, 'a') as output:
                    writer = csv.writer(output)
                    writer.writerow(['Initial', '', '', '', last_prob])

            for i, df_group in enumerate(evidence_groups):
                target_probs = self.update_and_return_target_belief(df_group, [target_node_id], clear_evidence=False, undo=True)
                prob = target_probs[target_node_id][target_outcome_id]
                if method == 'bayes':
                    val = log10((prob * (1-last_prob)) / (last_prob * (1-prob))) * 100
                else:
                    target_probs_list = list(target_probs[target_node_id].values())
                    last_probs_list = list(last_probs[target_node_id].values())
                    val = sum([target_probs_list[i] * log2(target_probs_list[i]/last_probs_list[i]) for i in range(len(last_probs_list))])

                if val > max_val:
                    max_val = val
                    max_val_group_idx = i
                    prob_sheet = prob

            df_best_group = evidence_groups.pop(max_val_group_idx)
            df_evidence = df_evidence.append(df_best_group, ignore_index=True)
                
            with open(output_file, 'a') as output:
                writer = csv.writer(output)
                if groups:
                    writer.writerow([df_best_group.loc[0, 'feature'], df_best_group.loc[0, 'value'], list(df_best_group['time']), max_val, prob_sheet])
                else:
                    writer.writerow([df_best_group.loc[0, 'feature'], list(df_best_group['value']), '0-9', max_val, prob_sheet])


    def most_relevant_explanation(self, output_file, targets, explanation_vars, time_limit=0):
        """Applies the MRE method of Yuan et al. 

        Args:
            output_file (str): path to which the output files should be saved
            targets (dict): the target variables and their chosen value
            explanation_vars (dict): the set of features that can be used in the explanations, not all are necessarily used in the most relevant explanation
            time_limit (int, optional): The number of time slice that should not be considered in the explanation. Defaults to 0 (consider every slice).
        """
        # [{'gbf' : gbf, 'vars' : [...], 'slices' : [(.,.),...], 'outcomes' : [...]}, ...]
        explns = []

        self.net.clear_all_evidence()
        for tar, tar_value in targets.items():
            # set value of target variables
            self.net.set_temporal_evidence(tar, self.get_slice_count()-1, tar_value)
        self.net.update_beliefs()
        # prior evidence (used to compute GBF)
        prob_tar = self.net.prob_evidence()

        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1)) 

        for var_subset in powerset(explanation_vars):
            # go through all combinations of consecutive time slices
            for time_assignment in itertools.product(*([list(itertools.product(*([list(range(self.get_slice_count()-time_limit))] * 2)))] * len(var_subset))):
                valid = True
                for e in time_assignment:
                    if e[0] > e[1]:
                        valid = False
                        break
                if not valid:
                    continue

                assert len(time_assignment) == len(var_subset)
                
                # got through all possible values of variables
                for outcomes in itertools.product(*([list(range(self.net.get_outcome_count(var))) for var in var_subset])):
                    assert len(outcomes) == len(var_subset)
                    outcomes = [self.net.get_outcome_id(var_subset[i], outcomes[i]) for i in range(len(outcomes))]

                    self.net.clear_all_evidence()

                    for var_idx in range(len(var_subset)):
                        for t in range(time_assignment[var_idx][0], time_assignment[var_idx][1]+1):
                            self.net.set_temporal_evidence(var_subset[var_idx], t, outcomes[var_idx])

                    # perform inference to compute posterior probability
                    self.net.update_beliefs()

                    # compute GBF
                    prediction_result = []
                    for tar, tar_value in targets.items():
                        outcome_count = self.net.get_outcome_count(tar)
                        outcome_names = self.net.get_outcome_ids(tar)
                        outcome_idx = outcome_names.index(tar_value)
                        node_probs = self.net.get_node_value(tar)
                        node_probs = node_probs[(self.get_slice_count()-1) * outcome_count : self.get_slice_count() * outcome_count]
                        prediction_result.append({tar : node_probs[outcome_idx]})

                    prob_var = self.net.prob_evidence()

                    for tar, tar_value in targets.items():
                        self.net.set_temporal_evidence(tar, self.get_slice_count()-1, tar_value)

                    self.net.update_beliefs()
                    prob_var_tar = self.net.prob_evidence()

                    gbf = ((prob_var_tar / prob_tar) * (1-prob_var)) / (prob_var * (1-(prob_var_tar/prob_tar)))
                
                    explns.append({'gbf' : gbf, 'vars' : var_subset, 'slices' : time_assignment, 'outcomes' : outcomes, 'predicition' : prediction_result})

        explns.sort(key=lambda e: e['gbf'], reverse=True)
        
        with open(output_file, 'w') as csvfile:
            fieldnames = ['GBF', 'Feature vars', 'Feature slices', 'Feature outcomes', 'Target prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(explns)):
                writer.writerow({'GBF' : explns[i]['gbf'], 'Feature vars' : explns[i]['vars'], 'Feature slices' : explns[i]['slices'], 'Feature outcomes' : explns[i]['outcomes'], 'Target prediction' : explns[i]['predicition']})





    # The following functions were used to test if information gain/KL divergence can be used to produce good explanations. I didn't include this in my thesis.


    def get_beliefs(self, target, time_slice):
        target_handle = self.net.get_node(target)
        outcome_count = self.net.get_outcome_count(target_handle)
        node_probs = self.net.get_node_value(target_handle)
        node_probs_subset = node_probs[time_slice * outcome_count : (time_slice + 1) * outcome_count]
        return {self.net.get_outcome_id(target_handle, outcome) : node_probs_subset[outcome] for outcome in range(outcome_count)}

    
    def get_entropy(self, target, time_slice=9, clear=True):
        if clear:
            self.net.clear_all_evidence()
            self.net.update_beliefs()
        outcome_count = self.net.get_outcome_count(target)
        node_probs = self.net.get_node_value(target)
        node_probs_subset = node_probs[time_slice * outcome_count : (time_slice + 1) * outcome_count]
        return -sum([node_probs_subset[i]*log2(node_probs_subset[i]) for i in range(len(node_probs_subset))])

    
    def get_cond_entropy(self, target, evidence):
        self.net.clear_all_evidence()
        self.net.clear_all_targets()
        self.net.set_target(target, True)
        self.net.set_target(evidence, True)
        self.net.update_beliefs()

        evidence_beliefs = []
        for time_slice in range(self.net.get_slice_count()): #
            if time_slice > 5: # here
                evidence_beliefs.append(self.get_beliefs(evidence, time_slice))
        print(evidence_beliefs)

        sum_list = []
        for outcomes in list(itertools.product(*evidence_beliefs)):
            # print(outcomes)
            val = 1
            val_str = '1'
            for time_slice, o in enumerate(outcomes):
                self.net.set_temporal_evidence(evidence, time_slice + 6, o) # here
                x = evidence_beliefs[time_slice][o]
                val *= x
                val_str += ' * ' + str(round(x, 4))
            self.net.update_beliefs()
            x = self.get_entropy(target, clear=False)
            val *= x
            val_str += ' * ' + str(round(x, 4))
            print(val_str)
            self.net.clear_all_evidence()
            sum_list.append(val)
        print(sum_list)
        return sum(sum_list)

        
    def information_gain(self, target, evidence, slices=list(range(10))):
        self.net.clear_all_evidence()
        self.net.clear_all_targets()
        self.net.set_target(target, True)

        outcome_count = self.net.get_outcome_count(evidence)
        # slice_count = self.get_slice_count()
        slice_count = len(slices)
        start_slice = slices[0]
        target_outcome_count = self.net.get_outcome_count(target)

        ret_value = 0

        # count = 0

        if outcome_count <= 3 or slice_count == 5:
            # print(evidence, outcome_count, slice_count)
            for outcomes in itertools.product(*([list(range(outcome_count))] * slice_count)):
                # count += 1
                # if count % 1000 == 0:
                #     print(count)
                for i, o in enumerate(outcomes):
                    time_slice = start_slice + i
                    self.net.set_temporal_evidence(evidence, time_slice, o)
                self.net.update_beliefs()
                node_probs = self.net.get_node_value(target)
                node_probs_subset = node_probs[(self.get_slice_count()-1) * target_outcome_count : (self.get_slice_count()) * target_outcome_count]
                entropy = -sum([node_probs_subset[i]*log2(node_probs_subset[i]) for i in range(len(node_probs_subset))])
                ret_value += self.net.prob_evidence() * entropy
                self.net.clear_all_evidence()
            return [ret_value]
        else:
            # print('else-case')
            ret1 = self.information_gain(target, evidence, slices=list(range(5)))
            # print('part1 done')
            ret2 = self.information_gain(target, evidence, slices=list(range(5,10)))
            # print('part2 done')
            return [ret1[0], ret2[0]]
