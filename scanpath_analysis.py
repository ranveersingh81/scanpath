
import pandas as pd
import os 
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
import pickle
import scanpath as scp


def get_file_data(file):
    file_path = os.path.join('./data/scanpaths/', file)
    df = pd.read_csv(file_path, delimiter='\t')
    cols = df.columns.to_list()
    df['file'] = file.replace("_scanpath.tsv", "")
    df = df[['file'] + cols]
    return df


files = [i for i in os.listdir('./data/scanpaths/')]
master_df = get_file_data(files[0])

for i  in  files[1:]:
    master_df = pd.concat([master_df, get_file_data(i)])

reader_data = pd.read_csv("/Users/RSingh81/Documents/personal_projects/scanpath/data/reader_meta_mean_and_per-text.csv")


all_columns = ['file', 'fixation_index', 'text_domain', 'trial', 'acc_bq_1', 'acc_bq_2', 'acc_bq_3', 'acc_tq_1', 'acc_tq_2', 'acc_tq_3',
'fixation_duration', 'next_saccade_duration', 'previous_saccade_duration', 'version', 'line', 'roi', 'char_index_in_line', 
'original_fixation_index', 'is_fixation_adjusted', 'reader_id', 'text_id', 'fixation_position_x', 'fixation_position_y',
'word_index_in_text', 'sent_index_in_text', 'char_index_in_text', 'word', 'character', 'text_id_numeric']



def get_scanpath(scan_record):
    scanpath = master_df[master_df['file'] == scan_record]
    scanpath = scanpath[['fixation_index', 'fixation_duration', 'next_saccade_duration', 'previous_saccade_duration', 'line', 'char_index_in_line',
                    'fixation_position_x', 'fixation_position_y', 'word_index_in_text', 'sent_index_in_text', 'char_index_in_text']]
    scanpath['change_in_word_flag'] = (scanpath['word_index_in_text'].diff() == 0).astype(int).shift(-1)
    scanpath['change_in_word'] = (scanpath['change_in_word_flag'] == 0).cumsum().shift(1).fillna(0)
    scanpath['same_word_next_saccade_duration'] = scanpath['change_in_word_flag']*scanpath['next_saccade_duration']
    scanpath['next_saccade_duration'] = scanpath['next_saccade_duration'] - scanpath['same_word_next_saccade_duration']
    scanpath['fixation_duration'] = scanpath['same_word_next_saccade_duration'] + scanpath['fixation_duration']
    scanpath = scanpath.drop('same_word_next_saccade_duration', axis =1)
    scanpath = scanpath.groupby(['change_in_word', 'word_index_in_text']).aggregate({'fixation_duration':'sum', 'next_saccade_duration':'sum', 'fixation_position_x':'mean', 'fixation_position_y':'mean'}).reset_index()
    scanpath['cum_fixation_duration'] = scanpath['fixation_duration'].cumsum().shift(1).fillna(0)
    scanpath = scanpath.reset_index()
    return scanpath



def get_output(filecomb, out_dict):
    print(filecomb)
    center_x = 840
    center_y = 525
    distance = 61
    unit = 0.0282
    scanpath1 = get_scanpath(filecomb[0])        
    scanpath2 = get_scanpath(filecomb[1])
    score, path, alignment, path_df  = scp.rscasim(scanpath1, scanpath2, center_x, center_y, distance, unit, modulator=0.83)
    out_dict[filecomb[0] + "_" + filecomb[1]] = (score, path, alignment, path_df)



if __name__ == "__main__":

    manager = mp.Manager()
    out_dict = manager.dict()
    files = master_df['file'].unique()
    texts = master_df['text_id'].unique()
    persons = master_df['reader_id'].unique()
    files = files
    files = [i for i in files if 'p0' in i]
    file_count = len(files)
    combinations = []
    for i in range(file_count):
        for j in range(file_count):
            if j <=  i:
                combinations.append((files[i], files[j]))
    process = []
    cnt = 0
    for comb in combinations:
        process.append(mp.Process(target=get_output, args=(comb,out_dict)))
        cnt+=1
        if cnt%9 == 0:
            for proc in process:
                proc.start()
            for proc in process:
                proc.join()
            for proc in process:
                proc.terminate()
            process.clear()

        with open('output.pickle', 'wb') as handle:
            pickle.dump(dict(out_dict), handle, protocol=pickle.HIGHEST_PROTOCOL)
