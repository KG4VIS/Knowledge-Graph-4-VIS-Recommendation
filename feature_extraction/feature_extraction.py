#!/usr/bin/env python
# coding: utf-8

# In[1]:


import traceback
import pandas as pd

import os
import json
from time import time, strftime

from collections import OrderedDict

from features.single_field_features import extract_single_field_features
from outcomes.field_encoding_outcomes import extract_field_outcomes
from general_helpers import load_raw_data, clean_chunk

import pickle
import numpy as np
from sklearn.model_selection import train_test_split


MAX_FIELDS = 25
total_charts = 0
charts_without_data = 0
chart_loading_errors = 0
feature_extraction_errors = 0
charts_exceeding_max_fields = 0
CHUNK_SIZE = 1000


# In[2]:


def extract_features_from_fields(fields, chart_obj={}, fid=None, num_fields=2):
    results = {}

#     feature_names_by_type = {
#         'basic': ['fid'],
#         'single_field': [],
#         'field_outcomes': []
#     }

    df_feature_tuples_if_exists = OrderedDict({'fid': fid})
    df_feature_tuples = OrderedDict({'fid': fid})
    df_outcomes_tuples = OrderedDict()

    single_field_features, parsed_fields = extract_single_field_features(
            fields, fid, MAX_FIELDS=MAX_FIELDS, num_fields=num_fields)

    df_field_level_features = []
    for i, f in enumerate(single_field_features):
        if f['exists']:
            df_field_level_features.append(f)
        
    
    results['df_field_level_features'] = df_field_level_features

    field_level_outcomes = extract_field_outcomes(chart_obj)
    results['field_outcomes'] = list(
        list(field_level_outcomes)[0].keys())
    results['df_field_level_outcomes'] = field_level_outcomes

    return results


# In[3]:


def extract_chunk_features(args):
    chunk = args['chunk']
    batch_num = args['batch_num']
    chunk_num = args['chunk_num']  # chunk=None, batch_num=0, chunk_num=0):

    df = clean_chunk(chunk)

    num_all_one_type = 0
    global feature_extraction_errors
    global charts_exceeding_max_fields

    chunk_outcomes = []
    chunk_field_level_features = []
    chunk_field_level_outcomes = []

    feature_names_by_type = {}

    start_time = time()

    # Per dataframe
    for chart_num, chart_obj in df.iterrows():
        fid = chart_obj.fid
        table_data = chart_obj.table_data

        absolute_chart_num = ((chunk_num - 1) * CHUNK_SIZE) + chart_num
        if absolute_chart_num % 100 == 0:
            print('[Batch %s / Chunk %s][%s] %.1f: %s %s' %
                  (batch_num, chunk_num, absolute_chart_num, time() -
                   start_time, fid, 'https://plot.ly/~{0}/{1}'.format(*fid.split(':'))))

        fields = table_data[list(table_data.keys())[0]]['cols']
        sorted_fields = sorted(fields.items(), key=lambda x: x[1]['order'])
        num_fields = len(sorted_fields)
        if num_fields > MAX_FIELDS:
            charts_exceeding_max_fields += 1
            continue

        try:
            extraction_results = extract_features_from_fields(
                sorted_fields, chart_obj=chart_obj, fid=fid, num_fields=num_fields)

            chunk_field_level_features.extend(
                extraction_results['df_field_level_features'])
            chunk_field_level_outcomes.extend(
                extraction_results['df_field_level_outcomes'])
        except Exception as e:
            print('Uncaught exception: {}'.format(e))
            traceback.print_tb(e.__traceback__)
            continue

    r = {
        'field_level_features': pd.DataFrame(chunk_field_level_features),
        'field_level_outcomes': pd.DataFrame(chunk_field_level_outcomes),
    }

    return r


# In[4]:


def write_batch_results(batch_results, features_dir_name, write_header=False):
    batch_field_level_features_dfs = []
    batch_field_level_outcomes_dfs = []

    for r in batch_results:
        if not r['field_level_features'].empty:
            batch_field_level_features_dfs.append(r['field_level_features'])
        if not r['field_level_outcomes'].empty:
            batch_field_level_outcomes_dfs.append(r['field_level_outcomes'])

    concatenated_results = {
        'field_level_features_df': pd.concat(batch_field_level_features_dfs, ignore_index=True) if batch_field_level_features_dfs else pd.DataFrame(),
        'field_level_outcomes_df': pd.concat(batch_field_level_outcomes_dfs, ignore_index=True) if batch_field_level_outcomes_dfs else pd.DataFrame()
    }
    
    for (k, v) in concatenated_results.items():
        output_file_name = os.path.join(features_dir_name, f"{k}.csv")
        v.to_csv(output_file_name, mode='a', index=False, header=write_header)


# In[6]:


base_dir = ".."

raw_df_chunks = load_raw_data(chunk_size=CHUNK_SIZE, data_file_name = f"../data/raw_data_all.csv")
if not os.path.exists(os.path.join(base_dir, 'features')):
    os.mkdir(os.path.join(base_dir, 'features'))

features_dir_name = os.path.join(base_dir, 'features')

if not os.path.exists(features_dir_name):
    os.mkdir(features_dir_name)

    
first_batch = True
start_time = time()

batch_results = []
for i, chunk in enumerate(raw_df_chunks):
    chunk_num = i + 1
    r = extract_chunk_features({
        'chunk': chunk,
        'chunk_num': chunk_num,
        'batch_num': 'NA'
    })
    batch_results.append(r)
    write_batch_results(
        batch_results,
        features_dir_name,
        write_header=first_batch)
    batch_results = []
    first_batch = False

print('Total time: {:.2f}s'.format(time() - start_time))


# In[7]:


df_o = pd.read_csv(f"../features/field_level_outcomes_df.csv")
df_f = pd.read_csv(f"../features/field_level_features_df.csv")

df_o_dedup = df_o.drop_duplicates('field_id')
df_f_dedup = df_f.drop_duplicates('field_id')

df_o_dedup_clean = df_o_dedup[df_o_dedup['trace_type']                              .isin(['bar', 'box', 'heatmap', 'histogram', 'line', 'scatter'])]
df_f_dedup_clean = df_f_dedup[df_f_dedup['exists']!='exists']

df_all = df_o_dedup_clean[['field_id', 'fid', 'trace_type', 'is_xsrc', 'is_ysrc']].merge(df_f_dedup_clean, on=['field_id', 'fid'])


with open("preserve_id_v4.pkl", 'rb') as f:
    preserved_id = pickle.load(f)
    
with open("feature_list_float_bool.pkl", 'rb') as f:
    feature_list = pickle.load(f)


list_dataset_split = train_test_split(preserved_id, train_size=0.7, test_size=0.3)
all_train_id, all_test_id    = list_dataset_split[0], list_dataset_split[1]

df_train = df_all[df_all['fid'].isin(all_train_id)]
df_test = df_all[df_all['fid'].isin(all_test_id)]

dict_encoding_trace_type = {
    'bar': 0,
    'box': 1,
    'heatmap': 2,
    'histogram': 3,
    'line':4,
    'scatter':5
    
}


# In[8]:


df_train = df_train[df_train['trace_type'].isin(dict_encoding_trace_type.keys())]
df_test = df_test[df_test['trace_type'].isin(dict_encoding_trace_type.keys())]


df_train['trace_type_n'] = df_train['trace_type'].apply(lambda x: dict_encoding_trace_type[x])
df_test['trace_type_n'] = df_test['trace_type'].apply(lambda x: dict_encoding_trace_type[x])



def is_x_or_y(is_xsrc, is_ysrc, fid):
    if is_xsrc and pd.isnull(is_ysrc): 
        return 'x'
    if is_ysrc and pd.isnull(is_xsrc): 
        return 'y'
    else: 
        return None

df_train['is_x_or_y'] = np.vectorize(is_x_or_y)(df_train['is_xsrc'], df_train['is_ysrc'], df_train['fid'])
df_train[feature_list['bool']] = df_train[feature_list['bool']].fillna(False)

def cut_off(x):
    if x >= quantile_1 and x <= quantile_3:
        return x
    elif x < quantile_1:
        return quantile_1
    else:
        return quantile_3

dict_cut_off = {}

for i in feature_list['float']:

    df_train[i] = np.array(df_train[i].astype('float32'))
    quantile_1 = np.quantile(df_train[i][np.isfinite(df_train[i])], 0.05)
    quantile_3 = np.quantile(df_train[i][np.isfinite(df_train[i])], 0.95)

    dict_cut_off[i] = (quantile_1, quantile_3)

    df_train[i] = df_train[i].apply(cut_off)


df_test['is_x_or_y'] = np.vectorize(is_x_or_y)(df_test['is_xsrc'], df_test['is_ysrc'], df_test['fid'])
df_test[feature_list['bool']] = df_test[feature_list['bool']].fillna(False)

for i in feature_list['float']:
    
    df_test[i] = np.array(df_test[i].astype('float32'))
    quantile_1, quantile_3 = dict_cut_off[i]
    df_test[i] = df_test[i].apply(cut_off)


df_train.to_csv((f"../features/feature_train.csv"), index=False)
df_test.to_csv((f"../features/feature_test.csv"), index=False)

