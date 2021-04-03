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

from sklearn.preprocessing import MinMaxScaler

embedding_path=f"../embeddings/TransE"
dict_path=f"../data/triplets"
dataset_path = f"../features"
inference_results_path = f"../inference_results"

with open(f"{dict_path}/test.pkl", "rb") as f:
    dict_data = pickle.load(f)

with open(f"../KG_construction/mdlp_discretization.json", "r") as f:
    dict_break_down = json.load(f)

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


prediction_class = ['line', 'scatter', 'bar', 'box', 'histogram', 'heatmap']
prediction_xy = ['is_x_src', 'is_y_src']
prediction_single = ['is_single_src_True', 'is_single_src_False']

dict_rules = {}
list_rules = []

for (k, v) in tqdm(dict_data.items()):
    # field_id, [[h1,r1], [h2,r2], ...]
    for (k_, v_) in v.items():
        # all rules
        list_embedding = []
        list_score_type = []
        list_score_xy = []
        for i in v_:
            # Generated rules already generated will not be generated again  
            try:             
                if (i[1], i[0]) not in dict_rules.keys():
                    # continuous rules
                    if i[0] in dict_break_down.keys():
                        index = int(i[1].split("_")[-1])
                        if index == 0:
                            range_ = f"(-inf, {round(dict_break_down[i[0]][0], 6)})"
                        elif index == len(dict_break_down[i[0]])-1:
                            range_ = f"({round(dict_break_down[i[0]][index-1], 6)}, +inf)"
                        else:
                            range_ = f"({round(dict_break_down[i[0]][index-1], 6)}, {round(dict_break_down[i[0]][index], 6)})"
                        
                        dict_rules[(i[1], i[0])] = {
                            "type": "quantitative",
                            "feature_value": range_
                        }
                        

                        embed = dict_entity_embedding[i[1]]+dict_relation_embedding[i[0]]
                        dict_score = {}
                    
                        for vis in prediction_class:
                            score = TransE(embed, dict_relation_embedding['trace_type'], dict_entity_embedding[vis],\
                                            gamma=config['gamma'])
                            dict_rules[(i[1], i[0])][vis] = score
                            dict_score[vis] = score
                            
                        scaler = MinMaxScaler()
                        score_normalized = scaler.fit_transform(np.array(list(dict_score.values())).reshape(-1, 1))
                        
                        for vis_idx in range(len(prediction_class)):
                            list_rules.append({
                                "feature_name": i[0],
                                "feature_value": range_,
                                "feature_value_all": dict_break_down[i[0]][:-1],
                                "feature_value_name": i[1],
                                "type": "continuous",
                                "vis": prediction_class[vis_idx],
                                "score": round(dict_score[prediction_class[vis_idx]], 6),
                                "norm_score": round(score_normalized[vis_idx][0], 6),
                            })

                    
                    # categorical rules
                    else:
                        dict_rules[(i[1], i[0])] = {
                            "type": "categorical",
                            "range": "N/A"
                        }

                        embed = dict_entity_embedding[i[1]]+dict_relation_embedding[i[0]]

                        dict_score = {}

                        for vis in prediction_class:
                            score = TransE(embed, dict_relation_embedding['trace_type'], dict_entity_embedding[vis],\
                                            gamma=config['gamma'])
                            dict_rules[(i[1], i[0])][vis] = score
                            dict_score[vis] = score


                        scaler = MinMaxScaler()
                        score_normalized = scaler.fit_transform(np.array(list(dict_score.values())).reshape(-1, 1))

                        for vis_idx in range(len(prediction_class)):
                            list_rules.append({
                                "feature_name": i[0],
                                "feature_value": i[1],
                                "feature_value_all": "N/A",
                                "feature_value_name": i[1],
                                "type": "categorical",
                                "vis": prediction_class[vis_idx],
                                "score": round(dict_score[prediction_class[vis_idx]], 6),
                                "norm_score": round(score_normalized[vis_idx][0], 6),
                            })
            except:
                continue


df = pd.DataFrame.from_dict(list_rules)
df.to_csv("../inference_results/rules.csv", index=False)
print(len(list_rules))
