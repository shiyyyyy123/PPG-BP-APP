import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import os
from configs import Config
from tqdm import tqdm
import json


def preprocess_data():
    dfs = []

    for i in range(5):
        df = pd.read_csv(f"data/raw/feat_fold_{i}.csv")
        df = df[(df['SP'].between(60, 200)) & (df['DP'].between(40, 120))]
        df = df.drop(['patient', 'trial'], axis=1, errors='ignore')
        dfs.append(df)

    full_df = pd.concat(dfs).reset_index(drop=True)
    print(f"Data loading completed. Total records: {len(full_df)}")

    imputer = KNNImputer(n_neighbors=7)
    full_df_imputed = pd.DataFrame(
        imputer.fit_transform(full_df),
        columns=full_df.columns
    )

    X = full_df_imputed.drop(['SP', 'DP'], axis=1)
    y_sp = full_df_imputed['SP']
    y_dp = full_df_imputed['DP']
    y = full_df_imputed[['SP', 'DP']]

    sp_selector = SelectFromModel(
        XGBRegressor(n_estimators=100, random_state=42),
        max_features=Config.max_features,
        threshold='median'
    )
    sp_selector.fit(X, y_sp)
    sp_selected_features = X.columns[sp_selector.get_support()]
    print(f"Systolic Pressure (SP) selected {len(sp_selected_features)} features")

    print("\nSelected features for Systolic Pressure (SP):")
    for i, feature in enumerate(sp_selected_features, 1):
        print(f"{i}. {feature}")

    sp_model = sp_selector.estimator_
    sp_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': sp_model.feature_importances_
    })
    sp_importances = sp_importances.sort_values('importance', ascending=False)
    sp_selected_importances = sp_importances[sp_importances['feature'].isin(sp_selected_features)]
    print("\nTop 10 important features for Systolic Pressure (SP):")
    print(sp_selected_importances.head(10).to_string(index=False))

    dp_selector = SelectFromModel(
        XGBRegressor(n_estimators=100, random_state=42),
        max_features=Config.max_features,
        threshold='median'
    )
    dp_selector.fit(X, y_dp)
    dp_selected_features = X.columns[dp_selector.get_support()]
    print(f"\nDiastolic Pressure (DP) selected {len(dp_selected_features)} features")

    print("\nSelected features for Diastolic Pressure (DP):")
    for i, feature in enumerate(dp_selected_features, 1):
        print(f"{i}. {feature}")

    dp_model = dp_selector.estimator_
    dp_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': dp_model.feature_importances_
    })
    dp_importances = dp_importances.sort_values('importance', ascending=False)
    dp_selected_importances = dp_importances[dp_importances['feature'].isin(dp_selected_features)]
    print("\nTop 10 important features for Diastolic Pressure (DP):")
    print(dp_selected_importances.head(10).to_string(index=False))

    selected_features = list(set(sp_selected_features) | set(dp_selected_features))
    print(f"\nMerged unique features count: {len(selected_features)}")

    MAX_FEATURES = Config.MAX_FEATURES

    if len(selected_features) > MAX_FEATURES:
        corr_sp = abs(X[selected_features].corrwith(y_sp))
        corr_dp = abs(X[selected_features].corrwith(y_dp))

        corr_avg = (corr_sp * Config.SP_power + corr_dp * (1 - Config.SP_power))

        selected_features = corr_avg.nlargest(MAX_FEATURES).index.tolist()
        print(f"After correlation-based selection, final features count: {len(selected_features)}")

        print("\nFinal selected features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")

        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'sp_correlation': [corr_sp.get(f, 0) for f in selected_features],
            'dp_correlation': [corr_dp.get(f, 0) for f in selected_features],
            'weighted_correlation': [corr_avg.get(f, 0) for f in selected_features]
        })
        feature_importance_df = feature_importance_df.sort_values('weighted_correlation', ascending=False)
        print("\nTop 10 correlated features:")
        print(feature_importance_df.head(10).to_string(index=False))

    X_selected = X[selected_features]

    print("Splitting data into train, validation and test sets...")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_selected, y, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.175, random_state=42  # 0.175 * 0.85 ≈ 0.15
    )

    print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")

    print("Performing data augmentation...")

    def augment_data(X, y, noise_level=0.03, n_augment=2):
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        if isinstance(y, pd.DataFrame):
            y_np = y.values
        else:
            y_np = y

        X_aug_list = [X_np]
        y_aug_list = [y_np]

        for i in range(n_augment):
            noise = np.random.normal(0, noise_level, X_np.shape)
            scales = np.random.uniform(0.95, 1.05, (X_np.shape[0], 1))
            X_aug_list.append(X_np * scales + noise)
            y_aug_list.append(y_np)

        if isinstance(y, pd.DataFrame):
            high_sp_mask = y['SP'] > 140
        else:
            high_sp_mask = y_np[:, 0] > 140

        if np.sum(high_sp_mask) > 0:
            X_high_sp = X_np[high_sp_mask]
            y_high_sp = y_np[high_sp_mask]

            noise = np.random.normal(0, noise_level * 0.8, X_high_sp.shape)
            scales = np.random.uniform(0.97, 1.03, (X_high_sp.shape[0], 1))
            X_aug_list.append(X_high_sp * scales + noise)
            y_aug_list.append(y_high_sp)

            noise2 = np.random.normal(0, noise_level * 0.6, X_high_sp.shape)
            scales2 = np.random.uniform(0.98, 1.02, (X_high_sp.shape[0], 1))
            X_aug_list.append(X_high_sp * scales2 + noise2)
            y_aug_list.append(y_high_sp)

        print(f"Original samples: {len(X)}, After augmentation: {sum(len(x) for x in X_aug_list)}")

        return np.vstack(X_aug_list), np.vstack(y_aug_list)

    X_train_aug, y_train_aug = augment_data(X_train, y_train, n_augment=3)
    print(f"Train set size after augmentation: {X_train_aug.shape}")

    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug.astype(np.float32))
    X_val_scaled = scaler.transform(X_val.values.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train_scaled)
    np.save("data/processed/X_val.npy", X_val_scaled)
    np.save("data/processed/X_test.npy", X_test_scaled)
    np.save("data/processed/y_train.npy", y_train_aug)
    np.save("data/processed/y_val.npy", y_val.values)
    np.save("data/processed/y_test.npy", y_test.values)
    np.save("data/processed/selected_features.npy", selected_features, allow_pickle=True)

    os.makedirs("data/processed/feature_info", exist_ok=True)

    sp_selected_importances.to_csv("data/processed/feature_info/sp_features.csv", index=False)
    dp_selected_importances.to_csv("data/processed/feature_info/dp_features.csv", index=False)
    pd.DataFrame({'feature': selected_features}).to_csv("data/processed/feature_info/final_features.csv", index=False)

    if 'feature_importance_df' in locals():
        feature_importance_df.to_csv("data/processed/feature_info/feature_correlations.csv", index=False)

    feature_stats = {
        "means": scaler.mean_.tolist(),
        "stds": np.sqrt(scaler.var_).tolist(),
        "feature_names": selected_features
    }

    with open("data/processed/feature_info/feature_stats.json", 'w') as f:
        json.dump(feature_stats, f, indent=2)

    print("Feature statistics saved to data/processed/feature_info/feature_stats.json")

    print(f"\nSelected {len(selected_features)} features")
    print(f"Final feature dimensions: {X_selected.shape[1]}")
    print(f"Train set size: {X_train_scaled.shape}")
    print(f"Validation set size: {X_val_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    print("Data preprocessing completed!")
    print(f"Feature information saved to data/processed/feature_info/ directory")


if __name__ == "__main__":
    preprocess_data()