"""
Takes in descriptor feature file and predicts
"""
import joblib
import pandas as pd
from pathlib import Path

label_field_to_exclude = [
    'tagg',
    'tm',
    'tmonset'
]

current_dir = Path(__file__).parent
model_dir = current_dir / "models"
print(model_dir)

model_map = {
    "tagg" : model_dir / "tagg/efs_best_knn.pkl",
    "tm" : model_dir / "tm/efs_best_randomforest.pkl",
    "tmon" : model_dir / "tmon/efs_best_elasticnet.pkl",
}

# data files that contain the features
data_files = {
    "tagg" : model_dir / "tagg/rf_efs.csv",
    "tm" : model_dir / "tm/rf_efs.csv",
    "tmon" : model_dir / "tmon/rf_efs.csv",
}


def build_model_feature_col_map():
    feature_col_map = {}
    for model, file in data_files.items():
        df = pd.read_csv(file)
        feature_col_map[model] = df.columns.tolist()[1:]
    return feature_col_map

# maps model to the columns that contain the features
model_feature_col_map = build_model_feature_col_map()
print(model_feature_col_map)

# Note: Holdout files are in the AbMelt directory for testing only
# In production, descriptors come from the pipeline
abmelt_data_dir = current_dir.parent / "AbMelt" / "data"
holdout_files = {
    "tagg" : abmelt_data_dir / "tagg/holdout.csv",
    "tm" : abmelt_data_dir / "tm/holdout.csv",
    "tmon" : abmelt_data_dir / "tmon/holdout.csv",
}

def infer_using_descriptors(model_name, descriptor_file, feature_names):
    model = joblib.load(model_map[model_name])
    df = pd.read_csv(descriptor_file)
    print(f"df shape: {df.shape}, columns: {df.columns}")
    df_features = df[feature_names]
    predictions = model.predict(df_features)
    return predictions

def main():
    for model_name, descriptor_file in holdout_files.items():
        feature_names = model_feature_col_map[model_name]
        feature_names = [feature_name for feature_name in feature_names if feature_name not in label_field_to_exclude]
        print(f"Model: {model_name}")
        print(f"Feature names: {feature_names}")
        # continue
        predictions = infer_using_descriptors(model_name, descriptor_file, feature_names)
        print(predictions)

if __name__ == "__main__":
    main()