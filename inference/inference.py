#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from copy import deepcopy
from time import time
from tqdm import tqdm
import numpy as np
import pickle
import json

embedding_path=f"../embeddings/TransE"
dict_path=f"../data/triplets"
dataset_path = f"../features"
inference_results_path = f"../inference_results"

with open(f"{dict_path}/test.pkl", "rb") as f:
    dict_data = pickle.load(f)

df_test = pd.read_csv(f"{dataset_path}/feature_test.csv")

entity_embedding = np.load(f"{embedding_path}/entity_embedding.npy")
relation_embedding = np.load(f"{embedding_path}/relation_embedding.npy")   

config =  json.load(open(f"{embedding_path}/config.json", 'r'))

with open(f"{dict_path}/entities.dict") as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(f"{dict_path}/relations.dict") as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)
        
dict_entity_embedding = {}
dict_relation_embedding = {}

for (k, v) in entity2id.items():
    dict_entity_embedding[k] = entity_embedding[v,:]
    
for (k, v) in relation2id.items():
    dict_relation_embedding[k] = relation_embedding[v,:]
    
del entity_embedding, relation_embedding



def TransE(head, relation, tail, gamma=24):

    score = (head + relation) - tail

    score = gamma - np.linalg.norm(score, ord=2)
    return score


dict_result = {}
prediction_class = ['line', 'scatter', 'bar', 'box', 'histogram', 'heatmap']
prediction_xy = ['is_x_src', 'is_y_src']
prediction_single = ['is_single_src_True', 'is_single_src_False']

for (k, v) in tqdm(dict_data.items()):
    # field_id, [[h1,r1], [h2,r2], ...]
    dict_result[k] = {}
    for (k_, v_) in v.items():
        list_score_type = []
        list_score_xy = []
        list_score_single = []
        
        for i in v_:
            try:
                embed = dict_entity_embedding[i[1]]+dict_relation_embedding[i[0]]

                list_score_type.append(np.array([TransE(embed,dict_relation_embedding['trace_type'], dict_entity_embedding[i],\
                                                        gamma=config['gamma']) for i in prediction_class]))
            
                list_score_xy.append(np.array([TransE(embed,dict_relation_embedding['is_x_or_y'],dict_entity_embedding[i],\
                                                      gamma=config['gamma']) for i in prediction_xy]))
            except:
                continue

        scores_each_class = np.mean(np.array(list_score_type),axis=0)
        scores_each_class_xy = np.mean(np.array(list_score_xy),axis=0)
        
        best_xy_index = np.argmax(np.array(scores_each_class_xy))

        L = list(np.argsort(np.array(scores_each_class)))
        
        dict_result[k][k_] = {
            "vis_score": scores_each_class,
            "matrix_scores": [prediction_class[i] for i in list(np.argsort(-np.array(scores_each_class)))],
            "best_class": prediction_class[L[-1]],
            "best_class_2": prediction_class[L[-2]],
            "xy_src": prediction_xy[best_xy_index].split("_")[1],
            "xy_score": scores_each_class_xy,
        }
        

def get_result(x):
    rank = dict_result[x['fid']][x['field_id']]['matrix_scores'].index(x['trace_type'])+1
    
    return dict_result[x['fid']][x['field_id']]['best_class'],     dict_result[x['fid']][x['field_id']]['best_class']==x['trace_type'] or dict_result[x['fid']][x['field_id']]['best_class_2']==x['trace_type'],    dict_result[x['fid']][x['field_id']]['xy_src'], rank, dict_result[x['fid']][x['field_id']]['best_class_2']


df_test[['predicted_class', 'result_check', 'predicted_xy', 'trace_rank', 'predicted_class_2']]    =df_test[['fid', 'field_id', 'trace_type']].apply(get_result, axis=1, result_type="expand")


from sklearn.metrics import accuracy_score

acc_top2 = sum(df_test['result_check'])/len(df_test)
acc_xy = accuracy_score(df_test['is_x_or_y'],df_test['predicted_xy'])
ave_rank = np.mean(df_test['trace_rank'])

print(f"Hits@2: {acc_top2}, XY Acc: {acc_xy}, Mean Rank: {ave_rank}")

df_test.to_csv(f'{dataset_path}/test_with_result_TransE.csv', index=False)
pickle.dump(dict_result, open(f"{inference_results_path}/TransE.pkl", "wb"))
