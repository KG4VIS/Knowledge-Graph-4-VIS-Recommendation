#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from copy import deepcopy
from time import time
from tqdm import tqdm
import numpy as np
import pickle


# In[2]:


with open(f"../feature_extraction/feature_list_float_bool.pkl", 'rb') as f:
    feature_list  = pickle.load(f)
    
with open("categorical_relation_mapping.pkl", 'rb') as f:
    dict_bool_type_special = pickle.load(f)
    
df_test = pd.read_csv("../features/feature_test.csv")


# In[3]:


df_test_dummy = deepcopy(df_test)

import json

with open(f"mdlp_discretization.json", "r") as f:
    dict_break_down = json.load(f)

for j in tqdm(feature_list['float']):
    
    if j not in dict_break_down.keys():
        continue

    df_test[j] = df_test[j].replace(np.nan, np.inf).replace(np.NINF, np.inf)

    df_test_dummy[f"{j}_digitalized"] =  np.digitize(df_test[j], bins=np.array(dict_break_down[j]), right=1)
    



for i in tqdm(feature_list['bool']):
    if i in dict_bool_type_special.keys():
        df_test_dummy[i] = df_test_dummy[i].apply(lambda x: dict_bool_type_special[i][1]
                                     if x == True else None)
    else:
        df_test_dummy[i] = df_test_dummy[i].apply(lambda x: f"{i}_{x}")


for i in tqdm(dict_break_down.keys()):  
    try:
        df_test_dummy[f"{i}_digitalized"] = df_test_dummy[f"{i}_digitalized"].apply(lambda x: 
                                                                i+"_"+str(x))
    except:
        print(i)
        continue

def get_feature_set(x):
    
    if x['fid'] not in dict_data.keys():
        dict_data[x['fid']] = {}
    
    
    list_h_r = []
    for i in df_test_dummy.columns:
        if i in feature_list['bool']: 
            if i in dict_bool_type_special.keys():
                if x[i] and x[i]!=None:
                    list_temp = deepcopy(dict_bool_type_special[i])
                    list_temp.reverse()
                    list_h_r.append(list_temp)                           
            else:
                list_h_r.append([i, x[i]])
        elif "digitalized" in i:
            if int(x[i].split("_")[-1]) < dict_max[i]:
                list_h_r.append([i.split("_digitalized")[0], x[i]])
            
            
    dict_data[x['fid']][x['field_id']] = list_h_r


# In[6]:


dict_data = {}

time_2 = time()

dict_max = {i:int(max(df_test_dummy[i]).split("_")[-1]) for i in df_test_dummy.columns if "digitalized" in i}

df_test_dummy.apply(get_feature_set, axis=1)

print("discretized continuous triplet generation time: "+ str(time()-time_2))


# In[7]:


time_3 = time()

with open("../data/triplets/test.pkl", "wb") as f:
    pickle.dump(dict_data, f)
    
print("test triplet save time: "+ str(time()-time_2))

