#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mdlp_v2 import MDLP2
import json
import pickle
import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from time import time


# In[2]:


with open(f"../feature_extraction/feature_list_float_bool.pkl", 'rb') as f:
    feature_list  = pickle.load(f)
    
with open("categorical_relation_mapping.pkl", 'rb') as f:
    dict_bool_type_special = pickle.load(f)
    
df_train = pd.read_csv("../features/feature_train.csv")


# In[3]:


df_train_dummy = deepcopy(df_train)

dict_break_down = {}

for j in tqdm(feature_list['float']):

    if j not in feature_list['float']:
        continue

    t = np.array(df_train[j])
    non_na_t = np.isfinite(t)
    label = np.array(df_train['trace_type_n'])[non_na_t]
    max_ = max(t[np.isfinite(t)])

    est = MDLP2(min_samples_split=int(0.1*len(t)), min_samples_leaf=int(0.05*len(t)))


    est.fit(t.reshape(-1,1), label)

    break_down = list(est.cut_points_[0]) + [float(max_+1)]

    dict_break_down[j] = break_down

    df_train[j] = df_train[j].replace(np.nan, np.inf).replace(np.NINF, np.inf)


    df_train_dummy[f"{j}_digitalized"] =  np.digitize(df_train[j],                                                        bins=np.array(break_down), right=1)

df_train_dummy = df_train_dummy.drop(feature_list['float'], axis=1)


with open(f"mdlp_discretization.json", "w") as f:
    json.dump(dict_break_down, f)


# In[4]:


bool_triplets = []

time_0 = time()

for i in tqdm(feature_list['bool']):
    if i in dict_bool_type_special.keys():
        df_train[[i, "field_id"]].apply(lambda x: bool_triplets.append(dict_bool_type_special[i]+[x['field_id']])                                     if x[i] == True else None, axis=1)
    else:
        df_train[[i, "field_id"]].apply(lambda x: bool_triplets.append([f"{i}_{x[i]}", i, x['field_id']]), axis=1)

print("categorical triplet generation time: "+str(time()-time_0))


# In[5]:


float_triplets = []

time_1 = time()

for i in tqdm(dict_break_down.keys()):
    try:
        max_ = max(df_train_dummy[f"{i}_digitalized"])

        df_train_dummy[[f"{i}_digitalized", "field_id"]].apply(lambda x: float_triplets.append([
                                                                i+"_"+str(x[f"{i}_digitalized"]), i,\
                            x['field_id']]) if x[f"{i}_digitalized"] < max_ else None, axis=1)
    except:
        continue

        
print("discretized continuous triplet generation time: "+ str(time()-time_1))


# In[6]:


vis_triplets = []

time_2 = time()


_ = df_train[["trace_type", "field_id"]].apply(lambda x: vis_triplets.append([x['field_id'], "trace_type",                                                                          x['trace_type']]), axis=1)

_ = df_train[df_train['is_x_or_y']!='None'][["is_x_or_y", "field_id"]].apply(lambda x: vis_triplets.append([x['field_id'], "is_x_or_y",                                                                          f"is_{x['is_x_or_y']}_src"]), axis=1)

print("vis triplet generation time: "+ str(time()-time_2))


# In[8]:


time_3 = time()

with open(f"../data/triplets/train.txt", "w") as f:
    for i in bool_triplets+vis_triplets+float_triplets:
        f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")
        

with open(f"../data/triplets/relations.dict", "w") as f:
    list_relations = set(list(np.array(bool_triplets+vis_triplets+float_triplets).T)[1])
    
    for i, element in enumerate(list_relations):
        f.write(f"{i}\t{element}\n")


with open(f"../data/triplets/entities.dict", "w") as f:
    list_entities = set(list(np.array(bool_triplets+vis_triplets+float_triplets).T)[0]).union(set(list(np.array(bool_triplets+vis_triplets+float_triplets).T)[2]))
    
    for i, element in enumerate(list_entities):
        f.write(f"{i}\t{element}\n")
        
print("triplet save time: "+ str(time()-time_2))

