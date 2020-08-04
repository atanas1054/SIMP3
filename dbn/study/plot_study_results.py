import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import numpy as np


def freq_real_plot(df):
    df_freq = df.filter(like=('frequency'))
    df_freq.columns = map(lambda col: col.replace('frequency', ''), df_freq.columns)
    df_real = df.filter(like=('realistic'))
    df_real.columns = map(lambda col: col.replace('realistic', ''), df_real.columns)

    # sns.set(style="whitegrid")
    sns.set()
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    sns.violinplot(data=df_freq, ax=axes[1], cut=0, scale='count', inner='stick')
    sns.violinplot(data=df_real, ax=axes[0], cut=0, scale='count', inner='stick')

    # axes[0].set_title(r"$\bf{How\ often\ do\ you\ see\ a\ scene\ like\ this\ on\ the\ street?}$")
    # axes[1].set_title(r"$\bf{How\ realistic\ is\ the\ scene\ for\ you?}$")
    axes[1].set_title("How often do you see a scene like this on the street?", fontsize=23)
    axes[0].set_title("How realistic is the scene for you?", fontsize=23)
    fig.tight_layout(pad=5.0)
    xticklabels = ['Sc1.1', 'Sc1.2', 'Sc1.3', 'Sc1.4', 'Sc1.5', 'Sc1.6', 'Sc2.1', 'Sc2.2', 'Sc3.1', 'Sc3.2', 'Sc3.4', 'Sc3.5', 'Sc3.6', 'Sc3.7', 'Sc3.8', 'Sc3.9', 'Sc3.10', 'Sc3.11', 'Sc4.1', 'Sc4.2', 'Sc4.3', 'Sc5.1', 'Sc5.2', 'Sc5.3', 'Sc5.4', 'Sc6.1', 'Sc7.1', 'Sc7.2']
    plt.setp(axes[1], xticklabels=xticklabels, yticks=[1, 2, 3, 4, 5], yticklabels=['Very rarely', 'Rarely', 'Medium', 'Often', 'Very often'])
    plt.setp(axes[0], xticklabels=xticklabels, yticks=[1, 2, 3, 4, 5], yticklabels=['Very unrealistic', 'Unrealistic', 'Neutral', 'Realistic', 'Very realistic'])
    plt.setp(axes[1].get_xticklabels(), fontsize=15, rotation=-45)
    plt.setp(axes[1].get_yticklabels(), fontsize=15)
    plt.setp(axes[0].get_xticklabels(), fontsize=15, rotation=-45)
    plt.setp(axes[0].get_yticklabels(), fontsize=15)

    plt.savefig('plots/freq_real_plot.pdf', bbox_inches='tight')
    plt.close('all')

def create_plot_premium(df_all, scene, df_cutpoints, df_avg):
    cut = df_cutpoints.loc[scene]
    p1 = cut['p1'].upper()
    p2 = cut['p2'].upper()
    p1_side = crossing_side[scene_codes.index(scene)]

    ## DF for violinplots
    df_all = df_all.filter(like='rating' + scene)
    df_dict = {'df_p1_imas' : df_all.filter(like='Imas' + p1).copy(), 'df_p1_wti' : df_all.filter(like='Wti' + p1).copy(), 'df_p2_imas' : df_all.filter(like='Imas' + p2).copy(), 'df_p2_wti' : df_all.filter(like='Wti' + p2).copy()}
    for key, df in df_dict.items():
        df.columns = map(lambda col_name: col_name[:col_name.find('[')], df.columns)
        df.loc[:,'index'] = df.index
        df = pd.wide_to_long(df, ['rating' + scene], i='index', j='cutpoints')
        if 'p1' in key:
            df.loc[:,'pedestrian'] = len(df.index) * ['P1']
        else:
            df.loc[:,'pedestrian'] = len(df.index) * ['P2']
        df_dict[key] = df
    df_both_imas = pd.concat([df_dict['df_p1_imas'], df_dict['df_p2_imas']])
    df_both_imas = df_both_imas.reset_index()
    df_both_wti = pd.concat([df_dict['df_p1_wti'], df_dict['df_p2_wti']])
    df_both_wti = df_both_wti.reset_index()

    ## DF for stripplot
    df_avg = df_avg.transpose()
    df_avg = df_avg.filter(like='rating' + scene)
    df_avg = df_avg.reset_index()
    df_avg_dict = {'Imas' : df_avg.filter(like='Imas').copy(), 'Wti' : df_avg.filter(like='Wti').copy()}
    for key, df in df_avg_dict.items():
        df['index'] = df.index
        df = pd.wide_to_long(df, 'rating' + scene, i='index', j='suffix', suffix='.+')
        df = df.reset_index()
        df['cutpoint'], df['pedestrian'] = df['suffix'].str.split('[', 1).str
        df = df.drop(['index', 'suffix'], axis=1)
        df['pedestrian'] = df['pedestrian'].apply(lambda e: e.replace(key + p1 + ']', 'P1'))
        df['pedestrian'] = df['pedestrian'].apply(lambda e: e.replace(key + p2 + ']', 'P2'))
        df['cutpoint'] = df['cutpoint'].astype(int)
        df_avg_dict[key] = df

    if p1_side == 'left':
        hue_order = ['P1', 'P2']
    else:
        hue_order = ['P2', 'P1']

    ## Plot IMAS
    sns.set() # Aussehen!
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].set_title(r"$\bf{IMAS}$")
    sns.violinplot(data=df_both_imas, x='cutpoints', y='rating' + scene, hue='pedestrian', hue_order=hue_order, ax=axes[0], cut=0, inner='stick', split=True, bw=1) # , palette="Set2"
    sns.stripplot(data=df_avg_dict['Imas'], x='cutpoint', y='rating' + scene, hue='pedestrian', hue_order=hue_order, ax=axes[0], jitter=False, linewidth=1.5, size=8) 
    axes[0].legend_.remove()
    # handles, labels = axes[0].get_legend_handles_labels()
    # labels = [labels[0], labels[1], labels[2] + ' avg', labels[3] + ' avg']
    # axes[0].legend(handles, labels)

    # Plot WTI
    axes[1].set_title(r"$\bf{WTI}$")
    sns.violinplot(data=df_both_wti, x='cutpoints', y='rating' + scene, hue='pedestrian', hue_order=hue_order, ax=axes[1], cut=0, inner='stick', split=True, bw=1)
    sns.stripplot(data=df_avg_dict['Wti'], x='cutpoint', y='rating' + scene, hue='pedestrian', hue_order=hue_order, ax=axes[1], jitter=False, linewidth=1.5, size=8)
    handles, _ = axes[1].get_legend_handles_labels()
    # Legende oben in einer Zeile:
    # fig.legend(handles, ['ped left', 'ped right', 'ped left avg', 'ped right avg'], loc='upper center', ncol=4)
    # axes[1].legend_.remove()
    # Legende rechts:
    axes[1].legend(handles, ['ped left', 'ped right', 'ped left avg', 'ped right avg'], loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
    

    xticklabels = []
    for index, value in cut.items():
        try:
            i = int(value) - 1
            if len(xticklabels) <= i:
                xticklabels.append(index)
            else:
                xticklabels[i] = xticklabels[i] + '\n' + index
        except ValueError:
            pass

    plt.setp(axes, yticks=[1, 2, 3, 4, 5], yticklabels=['Very low', 'Low', 'Medium', 'High', 'Very high']) # , ylim=(0.7,5.3), xticklabels=xticklabels
    axes[0].set_xlabel('Video cutpoints') # , fontweight='bold'
    axes[0].set_ylabel('Study answers')
    axes[1].set_xlabel('Video cutpoints')
    axes[1].set_ylabel('Study answers')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25)

    plt.savefig('plots/plot_' + scene + '.pdf')
    plt.close('all')


def results_scenarios2(df, scenario, df_cutpoints):
    data = {'imas' : pd.DataFrame(columns=['cut', 'val', 'pedestrian']),
            'wti' : pd.DataFrame(columns=['cut', 'val', 'pedestrian'])}
    # data = {'p1' : pd.DataFrame(columns=['cut', 'val', 'variable']),
    #         'p2' : pd.DataFrame(columns=['cut', 'val', 'variable'])}


    # cut_names_list = ['', 'start', 'p1_gaze', 'p1_gesture', 'p2_gaze', 'p2_gesture']

    for scene in scene_codes:
        if scene.startswith(scenario):
            row_cutpoints = df_cutpoints.loc[scene]
            p1 = row_cutpoints['p1'].upper()
            p2 = row_cutpoints['p2'].upper()

            df_subset = df.filter(like='rating' + scene)
            for col in df_subset.columns:
                cut_num = col[col.find('[')-1]
                for cut_name, value in row_cutpoints.items():
                    try:
                        value = str(int(value))
                    except ValueError:
                        continue
                    if value == cut_num:
                        break
                else:
                    assert False
                p = 'p1' if p1 in col else 'p2'
                target = 'imas' if 'Imas' in col else 'wti'
                # new_lines = pd.DataFrame({'cut' : [cut_name] * len(df_subset.index), 
                #                           'val' : list(df_subset[col]), 
                #                           'variable' : [target] * len(df_subset.index)})
                new_lines = pd.DataFrame({'cut' : [cut_name] * len(df_subset.index), 
                                          'val' : list(df_subset[col]), 
                                          'pedestrian' : [p] * len(df_subset.index)})
                # data[p] = pd.concat([data[p], new_lines], ignore_index=True)
                data[target] = pd.concat([data[target], new_lines], ignore_index=True)

    # data['p1']['val'] = data['p1'].apply(lambda row: row['val'] + np.random.uniform(low=-0.3, high=0.3, size=1), axis=1)
    # data['p2']['val'] = data['p2'].apply(lambda row: row['val'] + np.random.uniform(low=-0.3, high=0.3, size=1), axis=1)          
    data['imas']['val'] = data['imas'].apply(lambda row: row['val'] + np.random.uniform(low=-0.3, high=0.3, size=1), axis=1)
    data['wti']['val'] = data['wti'].apply(lambda row: row['val'] + np.random.uniform(low=-0.3, high=0.3, size=1), axis=1)
    

    sns.set() # Aussehen!
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    # axes[0].set_title('Crossing pedestrian')
    # axes[1].set_title('Non-crossing pedestrian')
    # sns.stripplot(x="cut", y="val", hue="variable", data=data['p1'], jitter=True, dodge=True, ax=axes[0], palette="Set2")
    # sns.stripplot(x="cut", y="val", hue="variable", data=data['p2'], jitter=True, dodge=True, ax=axes[1], palette="Set2")
    axes[0].set_title('IMAS')
    axes[1].set_title('WTI')
    sns.stripplot(x="cut", y="val", hue="pedestrian", data=data['imas'], order=['start', 'p1_gaze', 'p1_gesture', 'p2_gaze', 'p2_gesture'], jitter=True, dodge=True, ax=axes[0])
    sns.stripplot(x="cut", y="val", hue="pedestrian", data=data['wti'], order=['start', 'p1_gaze', 'p1_gesture', 'p2_gaze', 'p2_gesture'], jitter=True, dodge=True, ax=axes[1])


    plt.setp(axes, yticks=[1, 2, 3, 4, 5], yticklabels=['Very low', 'Low', 'Medium', 'High', 'Very high']) # , ylim=(0.7,5.3), xticklabels=xticklabels
    axes[0].set_xlabel('Video cutpoints') # , fontweight='bold'
    axes[0].set_ylabel('Study answers')
    axes[1].set_xlabel('Video cutpoints')
    axes[1].set_ylabel('Study answers')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25)

    plt.savefig('plots/plot_scenario_' + scenario + '.pdf')
    plt.close('all')

def results_scenarios(df, scenario_list, df_cutpoints):
    data = {'imas' : pd.DataFrame(columns=['cut', 'val', 'Pedestrian']),
            'wti' : pd.DataFrame(columns=['cut', 'val', 'Pedestrian'])}

    for scenario in scenario_list:
        for scene in scene_codes2:
            if scene.startswith(scenario):
                if scenario != '08':
                    row_cutpoints = df_cutpoints.loc[scene]
                    p1 = row_cutpoints['p1'].upper()
                    # p2 = row_cutpoints['p2'].upper()

                df_subset = df.filter(like='rating' + scene)
                for col in df_subset.columns:
                    cut_num = col[col.find('[')-1]
                    if scenario != '08':
                        if scenario != '06':
                            iter_object = reversed(row_cutpoints.keys())
                        else:
                            iter_object = row_cutpoints.keys()
                        for cut_name in iter_object:
                            value = row_cutpoints[cut_name]
                            try:
                                value = str(int(value))
                            except ValueError:
                                continue
                            if value == cut_num:
                                cut_name = cut_name.replace('_', ' ')
                                break
                        else:
                            assert False
                    else:
                        cut_name = cut_num
                    if scenario != '08' and scenario != '06':
                        p = 'crossing' if p1 in col else 'not crossing'
                    elif scenario == '08':
                        p = 'left' if 'F' in col else 'right'
                    else:
                        p = 'right' if 'F' in col else 'left'
                    target = 'imas' if 'Imas' in col else 'wti'
                    new_lines = pd.DataFrame({'cut' : [cut_name] * len(df_subset.index), 
                                            'val' : list(df_subset[col]), 
                                            'Pedestrian' : [p] * len(df_subset.index)})
                    data[target] = pd.concat([data[target], new_lines], ignore_index=True)
    

    data_amount = {'imas' : pd.DataFrame(columns=['cut', 'Pedestrian', 'amount']),
                   'wti' : pd.DataFrame(columns=['cut', 'Pedestrian', 'amount'])}

    pedestrians = ['crossing', 'not crossing']
    cuts = ['start', 'p1 gaze', 'p1 gesture', 'p2 gaze', 'p2 gesture']
    if '06' in scenario_list:
        assert len(scenario_list) == 1
        pedestrians = ['left', 'right']
    if '08' in scenario_list:
        assert len(scenario_list) == 1
        pedestrians = ['left', 'right']
        cuts = ['1', '2']

    
    for target in data:
        for cut in cuts:
            for ped in pedestrians:
                df = data[target]
                amount = len(df[(df['cut'] == cut) & (df['Pedestrian'] == ped)])
                if amount != 0:
                    data_amount[target] = data_amount[target].append({'cut' : cut, 'Pedestrian' : ped, 'amount' : amount}, ignore_index=True)

    print(scenario_list)
    print('imas')
    print(data_amount['imas'])
    print('wti')
    print(data_amount['wti'])

    order = ['start', 'p1 gaze', 'p1 gesture', 'p2 gaze', 'p2 gesture']
    hue_order = ['crossing', 'not crossing']
    if len(scenario_list) == 1:
        if scenario_list[0] == '02' or scenario_list[0] == '03':
            order.remove('start')
        if scenario_list[0] == '06':
            order.remove('p1 gaze')
            order.remove('p2 gaze')
            hue_order = ['left', 'right']
        if scenario_list[0] == '08':
            order = ['1', '2']
            hue_order = ['left', 'right']

    sns.set() # Aussehen!
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].set_title(r"$\bf{IMAS}$")
    axes[1].set_title(r"$\bf{WTI}$")
    sns.pointplot(x="cut", y="val", hue="Pedestrian", data=data['imas'], hue_order=hue_order, order=order, dodge=True, ax=axes[0], ci="sd")
    sns.pointplot(x="cut", y="val", hue="Pedestrian", data=data['wti'], hue_order=hue_order, order=order, dodge=True, ax=axes[1], ci="sd")

    plt.setp(axes, yticks=[1, 2, 3, 4, 5], yticklabels=['Very low', 'Low', 'Medium', 'High', 'Very high']) # , ylim=(0.7,5.3), xticklabels=xticklabels
    axes[0].set_xlabel('Video cutpoints') # , fontweight='bold'
    axes[0].set_ylabel('Study answers')
    axes[1].set_xlabel('Video cutpoints')
    axes[1].set_ylabel('Study answers')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25)

    if len(scenario_list) == 1:
        plt.savefig('plots/plot_scenario_' + scenario_list[0] + '.pdf')
    else:
        plt.savefig('plots/plot_scenarios_crossing.pdf')
    plt.close('all')
    # plt.show()


df = pd.read_csv('0-total/results-survey276257_fixed.csv')
# df_avg = pd.read_csv('0-total/results-survey276257_fixed_avg.csv', index_col='code')
df_cutpoints = pd.read_csv('cutpoints-critical.csv', index_col='scene')

# freq_real_plot(df)

scene_codes = ['0101', '0102', '0105', '0106', '0107', '0108', '0201', '0203', '03a02', '03a03', '03a04', '03a06', '03a07', '03a08', '03b05', '03b06', '03b07', '03b12', '0402', '0403', '0404', '0503', '0505', '0506', '0508', '0603']
scene_codes2 = ['0101', '0102', '0105', '0106', '0107', '0108', '0201', '0203', '03a02', '03a03', '03a04', '03a06', '03a07', '03a08', '03b05', '03b06', '03b07', '03b12', '0402', '0403', '0404', '0503', '0505', '0506', '0508', '0603', '0801', '0802']
crossing_side = ['left', 'left', 'left', 'right', 'left', 'left', 'left', 'right', 'left', 'left', 'right', 'right', 'right', 'left', 'left', 'left', 'left', 'left', 'left', 'right', 'left', 'right', 'right', 'left', 'right', '']
# for scene in scene_codes:
#     create_plot_premium(df, scene, df_cutpoints, df_avg)

# create_plot_premium(df, '0403', df_cutpoints, df_avg)

for scenario in ['01', '02', '03', '04', '05', '06', '08']:
    results_scenarios(df, [scenario], df_cutpoints)

results_scenarios(df, ['01', '02', '03', '04', '05'], df_cutpoints)
# results_scenarios(df, ['06'], df_cutpoints)