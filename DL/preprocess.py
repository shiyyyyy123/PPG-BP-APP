# preprocess.py 修复版
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

    
    # 数据加载与清洗
    dfs = []
    # 加载原始数据
    for i in range(5):
        df = pd.read_csv(f"data/raw/feat_fold_{i}.csv")
        df = df[(df['SP'].between(60, 200)) & (df['DP'].between(40, 120))]
        df = df.drop(['patient', 'trial'], axis=1, errors='ignore')
        dfs.append(df)

    full_df = pd.concat(dfs).reset_index(drop=True)
    print(f"加载完成，共有{len(full_df)}条记录")

    # 缺失值处理

    imputer = KNNImputer(n_neighbors=7)
    full_df_imputed = pd.DataFrame(
        imputer.fit_transform(full_df),
        columns=full_df.columns
    )

    # 特征工程
    X = full_df_imputed.drop(['SP', 'DP'], axis=1)
    y_sp = full_df_imputed['SP']  # 单独的收缩压标签
    y_dp = full_df_imputed['DP']  # 单独的舒张压标签
    y = full_df_imputed[['SP', 'DP']]

    # 特征选择 - 分别对SP和DP进行

    # 对收缩压(SP)进行特征选择
    sp_selector = SelectFromModel(
        XGBRegressor(n_estimators=100, random_state=42),
        max_features=Config.max_features,
        threshold='median'
    )
    sp_selector.fit(X, y_sp)
    sp_selected_features = X.columns[sp_selector.get_support()]
    print(f"收缩压(SP)选择了{len(sp_selected_features)}个特征")
    
    # 输出收缩压选择的特征
    print("\n收缩压(SP)选择的特征:")
    for i, feature in enumerate(sp_selected_features, 1):
        print(f"{i}. {feature}")
    
    # 获取并输出SP特征重要性
    sp_model = sp_selector.estimator_
    sp_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': sp_model.feature_importances_
    })
    sp_importances = sp_importances.sort_values('importance', ascending=False)
    sp_selected_importances = sp_importances[sp_importances['feature'].isin(sp_selected_features)]
    print("\n收缩压(SP)特征重要性TOP10:")
    print(sp_selected_importances.head(10).to_string(index=False))
    
    # 对舒张压(DP)进行特征选择
    dp_selector = SelectFromModel(
        XGBRegressor(n_estimators=100, random_state=42),
        max_features=Config.max_features,  # 减少为80个特征
        threshold='median'
    )
    dp_selector.fit(X, y_dp)
    dp_selected_features = X.columns[dp_selector.get_support()]
    print(f"\n舒张压(DP)选择了{len(dp_selected_features)}个特征")
    
    # 输出舒张压选择的特征
    print("\n舒张压(DP)选择的特征:")
    for i, feature in enumerate(dp_selected_features, 1):
        print(f"{i}. {feature}")
    
    # 获取并输出DP特征重要性
    dp_model = dp_selector.estimator_
    dp_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': dp_model.feature_importances_
    })
    dp_importances = dp_importances.sort_values('importance', ascending=False)
    dp_selected_importances = dp_importances[dp_importances['feature'].isin(dp_selected_features)]
    print("\n舒张压(DP)特征重要性TOP10:")
    print(dp_selected_importances.head(10).to_string(index=False))
    
    # 合并两个特征集并去重
    selected_features = list(set(sp_selected_features) | set(dp_selected_features))
    print(f"\n合并后共有{len(selected_features)}个唯一特征")
    
    # 最大特征数为100
    MAX_FEATURES = Config.MAX_FEATURES
    

    if len(selected_features) > MAX_FEATURES:
        # 计算特征与目标的相关性
        corr_sp = abs(X[selected_features].corrwith(y_sp))
        corr_dp = abs(X[selected_features].corrwith(y_dp))
        # 给收缩压相关性更高的权重
        corr_avg = (corr_sp * Config.SP_power + corr_dp * (1-Config.SP_power))  # 增加收缩压权重到75%
        # 选择相关性最高的MAX_FEATURES个特征
        selected_features = corr_avg.nlargest(MAX_FEATURES).index.tolist()
        print(f"通过相关性选择，最终保留{len(selected_features)}个特征")
        
        # 输出最终选择的特征
        print("\n最终选择的特征:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
            
        # 创建并保存特征相关性数据
        feature_importance_df = pd.DataFrame({
            'feature': selected_features,
            'sp_correlation': [corr_sp.get(f, 0) for f in selected_features],
            'dp_correlation': [corr_dp.get(f, 0) for f in selected_features],
            'weighted_correlation': [corr_avg.get(f, 0) for f in selected_features]
        })
        feature_importance_df = feature_importance_df.sort_values('weighted_correlation', ascending=False)
        print("\n最终特征的相关性TOP10:")
        print(feature_importance_df.head(10).to_string(index=False))

    
    # 构建特征子集
    X_selected = X[selected_features]

    # 三段式数据划分：训练集(70%)、验证集(15%)、测试集(15%)
    print("将数据划分为训练集、验证集和测试集...")
    # 先分离测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_selected, y, test_size=0.15, random_state=42
    )
    # 再将剩余数据分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.175, random_state=42  # 0.175 * 0.85 ≈ 0.15
    )

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")

    # 数据增强
    print("执行数据增强...")
    def augment_data(X, y, noise_level=0.03, n_augment=2):
        # 转换为numpy数组
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        

        if isinstance(y, pd.DataFrame):
            y_np = y.values
        else:
            y_np = y
            
        X_aug_list = [X_np]
        y_aug_list = [y_np]
        
        # 基本噪声和缩放增强
        for i in range(n_augment):
            # 噪声增强
            noise = np.random.normal(0, noise_level, X_np.shape)
            # 随机缩放
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
            
            # 增加额外的高血压样本增强
            noise2 = np.random.normal(0, noise_level * 0.6, X_high_sp.shape)
            scales2 = np.random.uniform(0.98, 1.02, (X_high_sp.shape[0], 1))
            X_aug_list.append(X_high_sp * scales2 + noise2)
            y_aug_list.append(y_high_sp)
            
        print(f"原始样本数: {len(X)}, 增强后: {sum(len(x) for x in X_aug_list)}")

        return np.vstack(X_aug_list), np.vstack(y_aug_list)


    X_train_aug, y_train_aug = augment_data(X_train, y_train, n_augment=3)
    print(f"增强后训练集大小: {X_train_aug.shape}")

    # 标准化
    print("标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug.astype(np.float32))
    X_val_scaled = scaler.transform(X_val.values.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

    # 保存数据
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train_scaled)
    np.save("data/processed/X_val.npy", X_val_scaled)
    np.save("data/processed/X_test.npy", X_test_scaled)
    np.save("data/processed/y_train.npy", y_train_aug)
    np.save("data/processed/y_val.npy", y_val.values)
    np.save("data/processed/y_test.npy", y_test.values)
    np.save("data/processed/selected_features.npy", selected_features, allow_pickle=True)
    
    # 将特征列表保存为CSV文件，便于查看
    os.makedirs("data/processed/feature_info", exist_ok=True)
    
    # 保存SP特征及其重要性
    sp_selected_importances.to_csv("data/processed/feature_info/sp_features.csv", index=False)
    
    # 保存DP特征及其重要性
    dp_selected_importances.to_csv("data/processed/feature_info/dp_features.csv", index=False)
    
    # 保存最终选择的特征及其相关性信息
    pd.DataFrame({'feature': selected_features}).to_csv("data/processed/feature_info/final_features.csv", index=False)
    
    if 'feature_importance_df' in locals():
        feature_importance_df.to_csv("data/processed/feature_info/feature_correlations.csv", index=False)
    
    # 导出特征标准化参数（均值和标准差）到JSON文件
    feature_stats = {
        "means": scaler.mean_.tolist(),
        "stds": np.sqrt(scaler.var_).tolist(),
        "feature_names": selected_features
    }
    
    with open("data/processed/feature_info/feature_stats.json", 'w') as f:
        json.dump(feature_stats, f, indent=2)
    
    print("特征统计信息已保存至 data/processed/feature_info/feature_stats.json")
    
    print(f"\n选择了{len(selected_features)}个特征")
    print(f"最终特征维度: {X_selected.shape[1]}")
    print(f"训练集大小: {X_train_scaled.shape}")
    print(f"验证集大小: {X_val_scaled.shape}")
    print(f"测试集大小: {X_test_scaled.shape}")
    print("数据预处理完成！")
    print(f"特征信息已保存至 data/processed/feature_info/ 目录")


if __name__ == "__main__":
    preprocess_data()
