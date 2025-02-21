from run_ml_algo_w_selected_features import r as r_selected
from run_ml_algo_wout_feature_select import r as r_noselected
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def extract_complete_features(r_score_original: dict[str, tuple[float, float]],
                              r_score_selected: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """
    Extract the complete features from the permutation importance dictionary.
    If a feature is missing in `r_score_selected`, it is assigned a value of 0.

    :param r_score_original: (dict[str, tuple[float, float]]) The full permutation importance dictionary;
    :param r_score_selected: (dict[str, tuple[float, float]]) The selected permutation importance dictionary;
    :return: (dict[str, float]) Dictionary with all features, missing ones filled with 0.
    """
    #print("Original features:", r_score_original.keys())
    #print("Selected features:", r_score_selected.keys())
    key_list_selected = {
        key: r_score_selected.get(key, (0.0, 0.0)) for key in r_score_original.keys()
    }

    return key_list_selected


if __name__ == "__main__":
    selected_keys = extract_complete_features(r_noselected, r_selected)

    selected_keys_correct_format = extract_complete_features(r_noselected, r_selected)
    plt.errorbar(selected_keys_correct_format.keys(), [x[0] for x in selected_keys_correct_format.values()], yerr=[x[1] for x in selected_keys_correct_format.values()], fmt='o', color="orangered",
                 label="Selected 8 features", capsize=6, capthick=2, elinewidth=2, errorevery=1)
    plt.errorbar(r_noselected.keys(), [x[0] for x in r_noselected.values()], yerr=[x[1] for x in r_noselected.values()], fmt='o', color="royalblue",
                 label="All features", capsize=6, capthick=2, elinewidth=2, errorevery=1)
    plt.xticks(rotation=45)
    plt.ylabel(r"$R$", fontdict={"fontsize": 16})
    plt.xlabel("Features", fontdict={"fontsize": 16})
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()