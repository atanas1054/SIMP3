# dbn = DbnWrapper('\\Users\\nomu01\\Documents\\Benchmark\\OpenDS-CTS02\\slices10\\01\\emidas-dbn-v9-10_trained.xdsl', trained=True)
# df = pd.read_csv('\\Users\\nomu01\\Documents\\Benchmark\\predictions\\01\\emidas-dbn-v9-50\\0101_horizontal-reflection_1_1\\dbn_prediction.csv', index_col=0)
# evidence_sheet(dbn, 'p1_imas', 'very_high', df.iloc[50:100], 'evidence_sheet.csv')
# entropy_sheet(dbn, 'p2_imas', 'entropy_sheet_p2_imas_sc_01_test.csv')


# dbn = DbnWrapper('input/emidas-dbn-v9-10_trained.xdsl', trained=True)
# df = pd.read_csv('input/dbn_prediction_0107_nr_1.csv', usecols=['p1_head', 'p2_head', 'p1_body', 'p2_body', 'p1_approach', 'p2_approach', 'p1_gesture', 'p2_gesture', 'dist'])
# row = 107  
# row_start = max(0, row - dbn.get_slice_count() + 2)
# print(df.iloc[row_start:row])
# evidence_sheet(dbn, 'p1_imas', 'very_high', df.iloc[row_start:row+1], 'evidence_sheet_no_group.csv', groups=False)
# evidence_sheet(dbn, 'p1_imas', 'very_high', df.iloc[row_start:row+1], 'evidence_sheet.csv' )