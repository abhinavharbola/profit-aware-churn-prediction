import shap


def get_tree_explainer(model):
    return shap.TreeExplainer(model)


def compute_shap_explanation(explainer, input_df, feature_names):
    shap_values = explainer.shap_values(input_df)
    return shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.values[0],
        feature_names=feature_names
    )
