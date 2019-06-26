from ml import model_objects, imputer, dataframe_column_labels

def get_model_dict(disease):
    model = model_objects[disease]
    feature_names = dataframe_column_labels.copy()
    label_index = feature_names.index(disease)
    del(feature_names[label_index])
    feature_means = imputer.initial_imputer_.statistics_.copy().tolist()
    del(feature_means[label_index])
    weights = model.coef_
    model_dict = dict()
    model_dict["intercept"] = model.intercept_[0]
    for i, feature_name in enumerate(feature_names):
        feature_dict = dict()
        feature_dict["coef"] = weights[0][i]
        feature_dict["mean"] = feature_means[i]
        model_dict[feature_name] = feature_dict
    return model_dict