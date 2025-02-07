from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
import pandas as pd


data, target = load_breast_cancer(return_X_y=True, as_frame=True)
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

new_data = pd.DataFrame(data=data_normalized, columns=data.columns)