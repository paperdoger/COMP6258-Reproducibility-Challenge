import numpy as np

import os
import matplotlib.pyplot as plt
import seaborn as sns





path='informer_ETTh1_ftMS_sl48_ll32_pl30_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
preds = np.load('./results/'+path+'/pred.npy')
trues = np.load('./results/'+path+'/true.npy')
metrics = np.load('./results/'+path+'/metrics.npy')





plt.figure(figsize=(12, 8))
plt.plot(trues[:,0,-1],  marker='o', linestyle='-', color='blue', label='True Values')
plt.plot(preds[:,0,-1], marker='x', linestyle='--', color='red', label='Predicted Values')
plt.title('Comparison of True Values and Predicted Values of informer')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.grid(True)
plt.show()

