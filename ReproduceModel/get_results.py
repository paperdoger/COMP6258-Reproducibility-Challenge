import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False


result_dir = 'results/long_term_forecast_test_none_TimeMixer_custom_sl96_pl1_dm128_nh8_el2_dl1_df512_fc1_ebtimeF_dtTrue_test_0'


pred = np.load(f'{result_dir}/pred.npy')  # shape: [samples, pred_len, variables]
true = np.load(f'{result_dir}/true.npy')  # shape: [samples, pred_len, variables]
print('pred.shape:', pred.shape)
print('true.shape:', true.shape)


df = pd.read_csv('dataset/ETT/ETTh1.csv')
cols = df.columns[1:]  
ot_index = list(cols).index('OT')


pred_values = pred[:, 0, ot_index]
true_values = true[:, 0, ot_index]


mae = mean_absolute_error(true_values, pred_values)
mse = mean_squared_error(true_values, pred_values)
rmse = sqrt(mse)
r2 = r2_score(true_values, pred_values)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

plt.figure(figsize=(10, 4))
plt.plot(true_values[:200], label='Ground Truth')
plt.plot(pred_values[:200], label='Prediction')
plt.title(f'Prediction vs Ground Truth')
plt.legend()
plt.tight_layout()
plt.savefig('etth1_prediction.png', dpi=300)
plt.show()


plt.figure
