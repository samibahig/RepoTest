# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:07:12 2021

@author: samib
"""

#VOTING/probability done by me:  3 best outputs of the 3 best CNN models: 
# We used the 3 best models of CNN, Model A, Model B, and Model C
# When Output model A  == Output Model B, it becomes the new output by a voting majority, otherwise we take the Output of Model C (accuracy 89.3) who has the best accuracy 

import pandas as pd
import numpy as np

ModelA = pd.read_csv('/content/sample_submission(89.1)').values[:,1]
ModelB = pd.read_csv('/content/sample_submission(89.122)').values[:,1]
ModelC = pd.read_csv('/content/sample_submission(89.300)').values[:,1]
pred = []
for i in range(60000):
    if  ModelA[i] == ModelB[i]:
        pred.append(ModelA[i])                                 
    else:
        pred.append(ModelC[i])
pred = pd.DataFrame(pred)
pred.to_csv('Modelsvoting_final(5).csv',index=True, index_label='Id', header=['Category'])