import pandas as pd
import csv
from numpy import mean
import numpy as np
import sys, re
from pathlib import Path

paths = {'total' : '0-total/results-survey276257.csv',
         'initial' : '1-initial/results-survey276257.csv',
         'extended' : '2-extended/results-survey276257.csv'}

cutpoint_dict = {}
# test_dict = {}
scene = ''
study_result_df = None
gt_var = ''


# def scene_compare(x, y):
#     # re.search(r'(\d{2}[ab]{0,1})', 'rating03a0101[ImasM]').group() => 03a
#     x_1 = re.search(r'(\d{2}[ab]{0,1})', x).group()
#     y_1 = re.search(r'(\d{2}[ab]{0,1})', y).group()
#     if x_1 != y_1:


def repair_study_results(file_path):
    """Deletes the uncompleted answer(s), repaires the codes and orders the scenes by scenarios. New file is saved with '_fixed' suffix.
    
    Arguments:
        file_path {str} -- Path to study file downloaded from LimeSurvey
    """
    df = pd.read_csv(file_path)
    df = df[df['lastpage'] == 98]
    col_new = []
    for col in df.columns:
        if 'WtiW' in col:
            col_new.append(col.replace('WtiW', 'WtiF'))
        elif '[Oft]' in col:
            col_new.append(col.replace('[Oft]', ''))
        elif '[Real]' in col:
            col_new.append(col.replace('[Real]', ''))
        else:
            col_new.append(col)
    df.columns = col_new
    reordered_cols = sorted(col_new[7:-4], key=lambda code: re.search(r'(\d{2}[ab]{0,1})', code).group())
    reordered_cols = sorted(reordered_cols, key=lambda code: re.search(r'\d{2}[ab]{0,1}(\d{2})', code).group())
    reordered_cols = col_new[:7] + reordered_cols + col_new[-4:]
    df = df[reordered_cols]
    df.to_csv(file_path[:-4] + '_fixed.csv', header=True, index=False)
    return file_path[:-4] + '_fixed.csv'


def compute_average(file_path):
    """Computes the average values of each question.
    
    Arguments:
        file_path {str} -- file path to survey csv file
    """
    df = pd.read_csv(file_path)
    df = df.filter(regex=("(?:rating|realistic|frequency).*"))
    means = df.mean(axis=0)
    means = means.to_frame(name='average')
    means = means.reset_index()
    means = means.rename(columns={'index' : 'code'})
    means.to_csv(file_path[:-4] + '_avg.csv', header=True, index=False)
    return file_path[:-4] + '_avg.csv'

def get_code_str(ped, cut):
    """Produces right question code for survey files.
    
    Arguments:
        ped {str} -- Must be either 'M' or 'F'
        cut {int} -- Cut number
    
    Returns:
        [type] -- [description]
    """
    if ped != 'M' and ped != 'F':
        sys.exit('Wrong argument in function "get_code_str": ', ped)
    return 'rating' + scene + '0' + str(int(cut)) + '[' + gt_var + ped + ']' # f.ex. 'rating010101[ImasF]'

def helper(cut_origin, to_ped, to_var, ):
    global cutpoint_dict
    # global test_dict
    code = get_code_str(ped=to_ped, cut=cut_origin)
    cutpoint_dict[to_var] = cutpoint_dict[to_var] + study_result_df[code].tolist()
    # test_dict[to_var].append(cut_origin)

def ground_truth_row(cuts):
    """[summary]
    
    Arguments:
        cuts {Series} -- Series giving which "cut-number" corresponds to which "event" (i.e., start, p1_gaze, ...)
    """
    p1 = cuts['p1'].upper()
    p2 = cuts['p2'].upper()

    global cutpoint_dict
    cutpoint_dict = {'p1_start' : [],
                     'p1_gaze' : [],
                     'p1_gesture' : [],
                     'p1_end' : [],
                     'p2_start' : [],
                     'p2_gaze' : [],
                     'p2_gesture' : [],
                     'p2_end' : []}

    # global test_dict
    # test _dict = {'p1_start' : [],
    #              'p1_gaze' : [],
    #              'p1_gesture' : [],
    #              'p1_end' : [],
    #              'p2_start' : [],
    #              'p2_gaze' : [],
    #              'p2_gesture' : [],
    #              'p2_end' : []}

    # Die cuts start/end sind "einfach": falls sie existieren, bekommen p1 und p2 den entsprechenden Wert für start/end zugewiesen
    for i in ['start', 'end']:
        cut = cuts[i]
        if not pd.isna(cut):
            helper(cut_origin=cut, to_ped=p1, to_var='p1_' + i)
            helper(cut_origin=cut, to_ped=p2, to_var='p2_' + i)

    # Der Rest ist nicht so einfach und keinesfalls auf einer übersichtlichen Art verallgemeinbar – daher leider alles einzeln
    p1_gaze_cut = cuts['p1_gaze']
    p1_gesture_cut = cuts['p1_gesture']
    p2_gaze_cut = cuts['p2_gaze']
    p2_gesture_cut = cuts['p2_gesture']

    # das muss eh in jedem Fall getan werden
    if not pd.isna(p1_gaze_cut):
        helper(cut_origin=p1_gaze_cut, to_ped=p1, to_var='p1_gaze')

    if not pd.isna(p1_gesture_cut):
        helper(cut_origin=p1_gesture_cut, to_ped=p1, to_var='p1_gesture')

    if not pd.isna(p2_gaze_cut):
        helper(cut_origin=p2_gaze_cut, to_ped=p2, to_var='p2_gaze')

    if not pd.isna(p2_gesture_cut):
        helper(cut_origin=p2_gesture_cut, to_ped=p2, to_var='p2_gesture')

    # 1 2 3 4
    if p1_gaze_cut + 3 == p2_gesture_cut:
        # Wert bei 'p1_gaze_cut' zu p2 start
        helper(cut_origin=p1_gaze_cut, to_ped=p2, to_var='p2_start')
        # Wert bei 'p1_gesture_cut' zu p2 start
        helper(cut_origin=p1_gesture_cut, to_ped=p2, to_var='p2_start')
        # Wert bei 'p2_gaze_cut' zu p1 end
        helper(cut_origin=p2_gaze_cut, to_ped=p1, to_var='p1_end')
        # Wert bei 'p2_gesture_cut' zu p1 end
        helper(cut_origin=p2_gesture_cut, to_ped=p1, to_var='p1_end')

    # na 1 2 3
    elif pd.isna(p1_gaze_cut) and p1_gesture_cut + 2 == p2_gesture_cut:
        # Wert bei 'p1_gesture_cut' zu p2 start
        helper(cut_origin=p1_gesture_cut, to_ped=p2, to_var='p2_start')
        # Wert bei 'p2_gaze_cut' zu p1 end
        helper(cut_origin=p2_gaze_cut, to_ped=p1, to_var='p1_end')
        # Wert bei 'p2_gesture_cut' zu p1 end
        helper(cut_origin=p2_gesture_cut, to_ped=p1, to_var='p1_end')

    # 1 2 na 3
    elif pd.isna(p2_gaze_cut) and p1_gaze_cut + 2 == p2_gesture_cut:
        # Wert bei 'p1_gaze_cut' zu p2 start
        helper(cut_origin=p1_gaze_cut, to_ped=p2, to_var='p2_start')
        # Wert bei 'p1_gesture_cut' zu p2 start
        helper(cut_origin=p1_gesture_cut, to_ped=p2, to_var='p2_start')
        # Wert bei 'p2_gesture_cut' zu p1 end
        helper(cut_origin=p2_gesture_cut, to_ped=p1, to_var='p1_end')

    # na 1 na 2
    elif pd.isna(p1_gaze_cut) and pd.isna(p2_gaze_cut) and p1_gesture_cut + 1 == p2_gesture_cut:
        # Wert bei 'p1_gesture_cut' zu p2 start
        helper(cut_origin=p1_gesture_cut, to_ped=p2, to_var='p2_start')
        # Wert bei 'p2_gesture_cut' zu p1 end
        helper(cut_origin=p2_gesture_cut, to_ped=p1, to_var='p1_end')

    # 1 2 2 3 / na 1 1 2
    elif p1_gesture_cut == p2_gaze_cut:
        # Wert bei 'p1_gaze_cut' zu p2 start
        if not pd.isna(p1_gaze_cut):
            helper(cut_origin=p1_gaze_cut, to_ped=p2, to_var='p2_start')
        # Wert bei 'p2_gesture_cut' zu p1 end
        helper(cut_origin=p2_gesture_cut, to_ped=p1, to_var='p1_end')

    # 1 2 na 2 / na 1 na 1
    elif pd.isna(p2_gaze_cut) and p1_gesture_cut == p2_gesture_cut:
        # Wert bei 'p1_gaze_cut' zu p2 start
        if not pd.isna(p1_gaze_cut):
            helper(cut_origin=p1_gaze_cut, to_ped=p2, to_var='p2_start')

    else:
        sys.exit('In the case differentiation a case has occurred which has not yet been treated.')
    

    # print(test_dict)
        
    row_dict = {}
    for key, value in cutpoint_dict.items():
        row_dict[key] = str(round(mean(value), 4))
    row_dict['scene'] = str(scene)

    return row_dict
    

    

def ground_truth_crit(study_path, cuts_filepath):
    global scene

    global study_result_df
    study_result_df = pd.read_csv(study_path)
    study_result_df = study_result_df.filter(regex=("rating.*"))

    cuts = pd.read_csv(cuts_filepath, index_col='scene')

    ## Ground truth IMAS
    global gt_var
    gt_var = 'Imas'

    idx_folder_end = study_path.find('/')
    gt_imas_path = study_path[:idx_folder_end] + '/ground_truth_imas'

    with open(gt_imas_path + '_v1.csv', 'w') as csvfile:
        fieldnames = ['scene', 'p1_start', 'p1_gaze', 'p1_gesture', 'p1_end', 'p2_start', 'p2_gaze', 'p2_gesture', 'p2_end']
        gt_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        gt_writer.writeheader()

        for cuts_index, cuts_row in cuts.iterrows():
            scene = cuts_index
            row = ground_truth_row(cuts_row)
            gt_writer.writerow(row)

    ## Ground truth WTI
    gt_var = 'Wti'

    gt_wti_path = study_path[:idx_folder_end] + '/ground_truth_wti'

    with open(gt_wti_path + '_v1.csv', 'w') as csvfile:
        fieldnames = ['scene', 'p1_start', 'p1_gaze', 'p1_gesture', 'p1_end', 'p2_start', 'p2_gaze', 'p2_gesture', 'p2_end']
        gt_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        gt_writer.writeheader()

        for cuts_index, cuts_row in cuts.iterrows():
            scene = cuts_index
            row = ground_truth_row(cuts_row)
            gt_writer.writerow(row)

    # Verbliebene NaN Felder:

    gt_imas_df = pd.read_csv(gt_imas_path + '_v1.csv', index_col='scene')
    gt_wti_df = pd.read_csv(gt_wti_path + '_v1.csv', index_col='scene')

    for gt_df in [gt_imas_df, gt_wti_df]:
        # p1 start
        val = gt_df.loc['0101', 'p1_start']
        for df_index, _ in gt_df.iterrows():
            if df_index[:2] == '01' or df_index[:2] == '02':
                gt_df.at[df_index, 'p1_start'] = val
        val = gt_df.loc['0503', 'p1_start']
        for df_index, _ in gt_df.iterrows():
            if df_index[:2] == '05':
                gt_df.at[df_index, 'p1_start'] = val
        val = gt_df.loc['0603', 'p1_start']
        for df_index, _ in gt_df.iterrows():
            if df_index[:2] == '03' or df_index == '0403' or df_index == '0404':
                gt_df.at[df_index, 'p1_start'] = val

        # p1 gaze
        val = (gt_df.loc['0107', 'p1_gaze'] + gt_df.loc['0203', 'p1_gaze']) / 2
        for df_index, _ in gt_df.iterrows():
            if (df_index[:2] == '01' or df_index[:2] == '02') and df_index != '0107' and df_index != '0203':
                gt_df.at[df_index, 'p1_gaze'] = val
        val = gt_df.loc['03b05', 'p1_gaze']
        for df_index, _ in gt_df.iterrows():
            if df_index[:2] == '03' or df_index == '0403' or df_index == '0404' or df_index == '0603':
                gt_df.at[df_index, 'p1_gaze'] = val
        val = (gt_df.loc['0503', 'p1_gaze'] + gt_df.loc['0508', 'p1_gaze']) / 2
        for df_index, _ in gt_df.iterrows():
            if df_index[:2] == '05' and df_index != '0503' and df_index != '0508':
                gt_df.at[df_index, 'p1_gaze'] = val

        # p2 start
        val = 0
        n = 0
        for df_index, _ in gt_df.iterrows():
            if df_index[:2] == '01' and not pd.isna(gt_df.loc[df_index, 'p2_start']):
                val = val + gt_df.loc[df_index, 'p2_start']
                n = n + 1
        gt_df.at['0105', 'p2_start'] = val / n
        gt_df.at['0108', 'p2_start'] = val / n
        val = 0
        n = 0
        for df_index, _ in gt_df.iterrows():
            if df_index[:3] == '03a' and not pd.isna(gt_df.loc[df_index, 'p2_start']):
                val = val + gt_df.loc[df_index, 'p2_start']
                n = n + 1
        gt_df.at['03a06', 'p2_start'] = val / n
        gt_df.at['03a07', 'p2_start'] = val / n
        gt_df.at['03a08', 'p2_start'] = val / n
        val = 0
        n = 0
        for df_index, _ in gt_df.iterrows():
            if df_index[:3] == '03b' and not pd.isna(gt_df.loc[df_index, 'p2_start']):
                val = val + gt_df.loc[df_index, 'p2_start']
                n = n + 1
        gt_df.at['03b06', 'p2_start'] = val / n
        gt_df.at['03b07', 'p2_start'] = val / n

        gt_df.at['0404', 'p2_start'] = gt_df.loc['0403', 'p2_start']

        val = 0
        n = 0
        for df_index, _ in gt_df.iterrows():
            if df_index[:2] == '05' and not pd.isna(gt_df.loc[df_index, 'p2_start']):
                val = val + gt_df.loc[df_index, 'p2_start']
                n = n + 1
        gt_df.at['0505', 'p2_start'] = val / n
        gt_df.at['0506', 'p2_start'] = val / n

        # p2 gaze
        for idx in gt_df.index:
            if pd.isnull(gt_df.at[idx, 'p2_gaze']):
                gt_df.at[idx, 'p2_gaze'] = (gt_df.at[idx, 'p2_start'] + gt_df.at[idx, 'p2_gesture']) / 2


    gt_imas_df = gt_imas_df.apply(lambda cell: round(cell,  2))
    gt_imas_df.to_csv(gt_imas_path + '_v2.csv')
    gt_wti_df = gt_wti_df.apply(lambda cell: round(cell,  2))
    gt_wti_df.to_csv(gt_wti_path + '_v2.csv')

    gt_imas_df = gt_imas_df.apply(lambda cell: round(cell,  0))
    gt_imas_df.to_csv(gt_imas_path + '_v3.csv')
    gt_wti_df = gt_wti_df.apply(lambda cell: round(cell,  0))
    gt_wti_df.to_csv(gt_wti_path + '_v3.csv')



def ground_truth_scenario_8(path_avg):
    path_prefix = path_avg.rsplit('/', 1)[0]
    df1 = pd.read_csv(path_avg, index_col='code')
    
    df2 = pd.DataFrame(np.array([['0801', df1.loc['rating080101[ImasF]', 'average'], df1.loc['rating080102[ImasF]', 'average'], df1.loc['rating080101[ImasM]', 'average'], df1.loc['rating080102[ImasM]', 'average']], ['0802', df1.loc['rating080201[ImasF]', 'average'], df1.loc['rating080202[ImasF]', 'average'], df1.loc['rating080201[ImasM]', 'average'], df1.loc['rating080202[ImasM]', 'average']]]), columns=['scene', 'p1_1', 'p1_2', 'p2_1', 'p2_2'])
    df2 = df2.set_index('scene')
    for col in df2.columns:
        df2[col] = df2[col].astype(float)
    df2 = df2.apply(lambda cell: round(cell,  0))
    df2.to_csv(path_prefix + '/sc_08_ground_truth_imas.csv')
    
    df3 = pd.DataFrame(np.array([['0801', df1.loc['rating080101[WtiF]', 'average'], df1.loc['rating080102[WtiF]', 'average'], df1.loc['rating080101[WtiM]', 'average'], df1.loc['rating080102[WtiM]', 'average']], ['0802', df1.loc['rating080201[WtiF]', 'average'], df1.loc['rating080202[WtiF]', 'average'], df1.loc['rating080201[WtiM]', 'average'], df1.loc['rating080202[WtiM]', 'average']]]), columns=['scene', 'p1_1', 'p1_2', 'p2_1', 'p2_2'])
    df3 = df3.set_index('scene')
    for col in df3.columns:
        df3[col] = df3[col].astype(float)
    df3 = df3.apply(lambda cell: round(cell,  0))
    df3.to_csv(path_prefix + '/sc_08_ground_truth_wti.csv')



def create_result_csv(directory):
    df = pd.read_csv(directory / 'results-survey276257_fixed.csv', usecols=(lambda col : 'rating' in col))
    
    df_std = pd.DataFrame(df.std(axis=0))
    df_std = df_std.rename_axis('code').reset_index()
    df_std['scene'] = df_std.apply(lambda row: row['code'][6:-7] if 'Imas' in row['code'] else row['code'][6:-6], axis=1)
    df_std['col'] = df_std.apply(lambda row: row['code'][-6:-1] + ' std' if 'Imas' in row['code'] else row['code'][-5:-1] + ' std', axis=1)
    df_std = df_std.drop(columns=['code'])
    df_std = df_std.rename(columns={0 : 'val'})

    df_mean = pd.DataFrame(df.mean(axis=0))
    df_mean = df_mean.rename_axis('code').reset_index()
    df_mean['scene'] = df_mean.apply(lambda row: row['code'][6:-7] if 'Imas' in row['code'] else row['code'][6:-6], axis=1)
    df_mean['col'] = df_mean.apply(lambda row: row['code'][-6:-1] + ' mean' if 'Imas' in row['code'] else row['code'][-5:-1] + ' mean', axis=1)
    df_mean = df_mean.drop(columns=['code'])
    df_mean = df_mean.rename(columns={0 : 'val'})

    df_both = pd.concat([df_mean, df_std], ignore_index=True)

    df_both = df_both.pivot(index='scene', columns='col', values='val')

    df_both = df_both.apply(lambda cell: round(cell,  2))

    renaming_dict = {'0101' : ('0101','M'),
                     '0102' : ('0102','F'), 
                     '0105' : ('0103','F'), 
                     '0106' : ('0104','M'), 
                     '0107' : ('0105','F'), 
                     '0108' : ('0106','F'), 
                     '0201' : ('0201','M'), 
                     '0203' : ('0202','F'), 
                     '03a02' : ('0301','F'), 
                     '03a03' : ('0302','F'), 
                     '03a04' : ('0303','M'), 
                     '03a06' : ('0304','M'), 
                     '03a07' : ('0305','M'), 
                     '03a08' : ('0306','F'), 
                     '03b05' : ('0307','M'), 
                     '03b06' : ('0308','F'), 
                     '03b07' : ('0309','F'), 
                     '03b12' : ('0310','F'), 
                     '0402' : ('0401','M'), 
                     '0403' : ('0402','F'), 
                     '0404' : ('0403','M'), 
                     '0503' : ('0501','M'), 
                     '0505' : ('0502','M'), 
                     '0506' : ('0503','F'), 
                     '0508' : ('0504','M'), 
                     '0603' : ('0601',''), 
                     '0801' : ('0801',''), 
                     '0802' : ('0802','')}
    df_both['scene'] = [x[:-2] for x in df_both.index] 
    df_both['cut'] = [x[-1] for x in df_both.index]
    df_both['crossing'] = df_both.apply(lambda row: renaming_dict[row['scene']][1], axis=1)
    df_both['scene'] = df_both.apply(lambda row: renaming_dict[row['scene']][0], axis=1)
    df_both = df_both.set_index(['scene', 'crossing', 'cut'])
    del df_both.columns.name
    
    df_both.to_csv(directory / 'study-result.csv')


# for _, path in paths.items():
#     fixed_path = repair_study_results(path)
#     path_avg_file = compute_average(fixed_path)
#     ground_truth_crit(fixed_path, 'cutpoints-critical.csv')
#     ground_truth_scenario_8(path_avg_file)


# ground_truth_scenario_8('/Users/nora/Documents/Uni/mastersthesis/emidas-code/study/0-total/results-survey276257_fixed_avg.csv')
# ground_truth_scenario_8('/Users/nora/Documents/Uni/mastersthesis/emidas-code/study/1-initial/results-survey276257_fixed_avg.csv')
# ground_truth_scenario_8('/Users/nora/Documents/Uni/mastersthesis/emidas-code/study/2-extended/results-survey276257_fixed_avg.csv')

create_result_csv(Path('/Users/nora/Documents/Uni/mastersthesis/emidas-code/study/0-total/'))
create_result_csv(Path('/Users/nora/Documents/Uni/mastersthesis/emidas-code/study/1-initial/'))
create_result_csv(Path('/Users/nora/Documents/Uni/mastersthesis/emidas-code/study/2-extended/')) 