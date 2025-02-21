from sklearn.ensemble import RandomForestClassifier
from data_prep import new_data, target
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
from run_ml_algo_w_selected_features import train, score, perm_importance
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
dataset_arr = scaler.fit_transform(new_data)

X_train, X_test_dummy, y_train, y_test_dummy = train_test_split(dataset_arr, target, test_size=0.5,
                                                                random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_dummy, y_test_dummy, test_size=0.33, random_state=42)
clf = RandomForestClassifier(max_depth=2, random_state=42)

preds = train(clf, X_train, y_train, X_val)
#print("Validation score:", score(preds, y_val))

r = perm_importance(clf, X_val, y_val, new_data)