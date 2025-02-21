import pandas as pd
import json


## suppose this is a json file

def select_dataframe(original_data: pd.DataFrame, jfile_path: str, visualize: bool = False):
    """
    Select the dataframe based on the selected features by the annealer.
    :param jfile_path: (str) The path to the json file containing the selected features;
    :param visualize: (bool) If True, print the 5 elements of the selected dataframe;
    :return: datafr: (pd.DataFrame) The selected dataframe.
    """
    with open(jfile_path, 'r') as fp:
        sol = json.load(fp)

    features = [key for key in sol.keys() if sol[key] == 1]
    index = [int(str(x)[2 :-1]) for x in features]
    selected_features = [original_data.columns.tolist()[i] for i in index]
    datafr = original_data[selected_features]

    if visualize:
        print(datafr.head())
    else:
        return datafr
