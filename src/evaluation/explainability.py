import numpy as np
import shap


def get_tree_explainer(model):
    return shap.TreeExplainer(model)


def compute_shap_explanation(explainer, input_df, feature_names):
    shap_values = explainer.shap_values(input_df)
    base_value = float(np.ravel(explainer.expected_value)[0])
    return shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=input_df.values[0],
        feature_names=feature_names
    )
