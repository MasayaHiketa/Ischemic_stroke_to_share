
########################################################################################
#Matching age+-2
########################################################################################
########################################################################################
print('-' * 100)

import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Precision loss occurred in moment calculation due to catastrophic cancellation.*"
)

# データ読み込み
hrv_df = pd.read_csv("extracted_data_2_4.csv")
icd_df = pd.read_csv("20241206_010627mimic-iii-wave-match-BasicTable-icd.csv")

# HRV対象の一意な subject_id でフィルタ
target_subjects = hrv_df['subject_id'].unique()
filtered_icd = icd_df[icd_df['subject_id'].isin(target_subjects)].drop_duplicates(subset='subject_id')
filtered_icd['age'] = filtered_icd['age'].clip(upper=89)
# 死亡・生存に分ける
dead_df = filtered_icd[filtered_icd['death'] == 1]
alive_df = filtered_icd[filtered_icd['death'] == 0]

matched_rows = []

# 各死亡者に対してマッチング
for _, row in dead_df.iterrows():
    age = row['age']

    # マッチ条件：年齢±1歳 & 性別一致
    candidates = alive_df[
        (alive_df['age'] >= age - 2) &
        (alive_df['age'] <= age + 2)
    ]

    # 最大5人まで
    matched_candidates = candidates.sample(n=min(len(candidates), 100), random_state=42)

    # ラベル付けして保存
    row_copy = row.copy()
    row_copy['matched_type'] = 'dead'
    matched_rows.append(row_copy)

    for _, match_row in matched_candidates.iterrows():
        match_copy = match_row.copy()
        match_copy['matched_type'] = 'alive'
        matched_rows.append(match_copy)

# DataFrame に変換して保存
matched_df = pd.DataFrame(matched_rows)
matched_df.to_csv("extract_data_2_4_matached.csv", index=False)
print("✅ Saved Match data: extract_data_2_4_matached.csv")
print('-' * 100)

########################################################################################
#Checking Matched dataset p-value
########################################################################################
########################################################################################
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# CSV読み込み
hrv_df = pd.read_csv("extract_data_2_4_matached.csv")
icd_df = pd.read_csv("20241206_010627mimic-iii-wave-match-BasicTable-icd.csv")

# HRVの対象患者（ユニーク）でフィルター
target_subjects = hrv_df['subject_id'].unique()
filtered_icd = icd_df[icd_df['subject_id'].isin(target_subjects)].drop_duplicates(subset=['subject_id'])
filtered_icd['age'] = filtered_icd['age'].clip(upper=89)

# 基本統計
total_patients = len(filtered_icd)
death_rate = filtered_icd['death'].mean()
num_deaths = filtered_icd['death'].sum()

print("Static Analysis")
print(f"Total Patients: {total_patients}")
print(f"Non-survived: {num_deaths}（Death Rate: {death_rate:.2%}）\n")
print('-' * 100)
# 年齢統計（生死別）
print("Age Statistics by Survival Status:")
print(filtered_icd.groupby('death')['age'].describe(), "\n")
# 年齢のp値（t検定）
age_p = ttest_ind(
    filtered_icd[filtered_icd['death'] == 0]['age'],
    filtered_icd[filtered_icd['death'] == 1]['age'],
    equal_var=False
).pvalue
print(f"Relationship between age and mortality (t-test p-value): {age_p:.4f}\n")
print('-' * 100)
# 性別分布（クロス集計）
print("Number of deaths vs. survivals by gender:")
gender_death = pd.crosstab(filtered_icd['gender'], filtered_icd['death'])
print(gender_death, "\n")
print('-' * 100)
# 性別ごとの年齢平均
print("Mean age by gender:")
print(filtered_icd.groupby('gender')['age'].mean(), "\n")
# 性別と死亡の関係（カイ二乗検定）
chi2, p_gender, _, _ = chi2_contingency(gender_death)
print(f"Association between gender and mortality (chi-square test p-value): {p_gender:.4f}")
print('-' * 100)

########################################################################################
#Take average of 5min-windows (windows number is range(18,24))
########################################################################################
########################################################################################

import pandas as pd

# === 1. ファイル読み込み ===
file_path = "extracted_data_2_4.csv"
df = pd.read_csv(file_path)

# === 2. 数値列だけを対象に抽出 ===
numeric_cols = df.select_dtypes(include='number').columns
feature_cols = [col for col in numeric_cols if col != 'subject_id']

# === 3. 各集約統計量を subject_id ごとに計算 ===
df_mean = df.groupby('subject_id')[feature_cols].mean()
#df_median = df.groupby('subject_id')[feature_cols].median().add_prefix('median_')
#df_std = df.groupby('subject_id')[feature_cols].std().add_prefix('std_')

# === 4. 結合 ===
df_stats = pd.concat([df_mean], axis=1).reset_index()

df= df.drop_duplicates(subset='subject_id', keep='first')
# === 5. 保存 ===
output_path = "hrv_features_window_avg_2_4.csv"
df_stats.to_csv(output_path, index=False)

# 結果の出力パス
output_path

# 行数確認
#print(f"✅ マージ完了: {df.shape[0]} 件の患者, {df.shape[1]} 列（行数変わらず）")




########################################################################################
#Select Features p <0.001
########################################################################################
########################################################################################
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# === データ読み込み（すでに患者ごとに1行） ===
hrv_df = pd.read_csv("hrv_features_window_avg_2_4.csv")
icd_df = pd.read_csv("extract_data_2_4_matached.csv")
merged = pd.merge(hrv_df, icd_df[['subject_id', 'death']], on='subject_id', how='inner').dropna(subset=['death'])


merged= merged.drop_duplicates(subset='subject_id', keep='first')
# 行数確認
print(f"✅ after merged: {merged.shape[0]} 件の患者, {merged.shape[1]} 列")
print('-' * 100)
# === 数値特徴量のうち 'erakde' を除外して p値フィルタリング ===
feature_cols = [col for col in merged.columns if col not in ['window','subject_id', 'death'] and np.issubdtype(merged[col].dtype, np.number)]
#pval = [(col, ttest_ind(merged[merged['death']==0][col], merged[merged['death']==1][col], equal_var=False)[1]) for col in feature_cols if 'erakde_b1.5_f0.5' in col.lower() and not col.lower().startswith('pct_')]
pval = [(col, ttest_ind(merged[merged['death']==0][col], merged[merged['death']==1][col], equal_var=False)[1]) for col in feature_cols if 'erakde'not in col.lower()]

pval_df = pd.DataFrame(pval, columns=['feature', 'p_value']).sort_values('p_value')
selected_features = pval_df[pval_df['p_value'] < 0.005]['feature'].tolist()
# === 抜く ===
exclude = [
    'window_size',
    'rrIntv_quantile50',
    'rrIntv_quantile75',
    'rrIntv_quantile25',
    'rrIntv_mean',
    'rrIntv_std',
    'rrIntv_percentile50'
]

selected_features = [f for f in selected_features if f not in exclude]
# 同じ値を持つ列を探して、どちらかを除く
dup_cols = []
for i, col1 in enumerate(selected_features):
    for col2 in selected_features[i+1:]:
        if np.allclose(merged[col1], merged[col2], equal_nan=True):
            dup_cols.append(col2)

# 重複削除（最初の1つだけ残す）
selected_features = [f for f in selected_features if f not in dup_cols]

selected_features = selected_features[:10]

# === 追加したい特徴量リスト（手動指定） ===
additional_features = [
    #'PCT_erakde_b1.5_f0.5_max_gt_5000',
    #'PCT_erakde_b1.2_f0.3_max_le_800',
    #'erakde_b1.5_f0.5_mean',
    #'erakde_b1.5_f0.5_std',
    #'erakde_b1.5_f0.5_max',
]


# === selected_features に追加（重複しないように） ===
selected_features = list(set(selected_features + additional_features))

# === 各特徴の生存・死亡ごとの平均±標準偏差とp値をまとめる ===
summary_data = []
for col in selected_features:
    mean_alive = merged[merged['death'] == 0][col].mean()
    std_alive = merged[merged['death'] == 0][col].std()
    mean_dead = merged[merged['death'] == 1][col].mean()
    std_dead = merged[merged['death'] == 1][col].std()

    # pval_df or erakde_pval_df のどちらかにあるか確認
    if col in pval_df['feature'].values:
        p_value = pval_df[pval_df['feature'] == col]['p_value'].values[0]
    elif col in erakde_pval_df['Feature'].values:
        p_value = erakde_pval_df[erakde_pval_df['Feature'] == col]['p-value'].values[0]
    else:
        p_value = np.nan  # 見つからなければNaN

    summary_data.append({
        'Feature': col,
        'P-value': p_value,
        'Survived(194)': f"{mean_alive:.4f} ± {std_alive:.4f}",
        'non-Survived(34)': f"{mean_dead:.4f} ± {std_dead:.4f}"
    })


summary_df = pd.DataFrame(summary_data)
# === p-value で昇順ソート（NaNは最後に） ===
summary_df = summary_df.sort_values(by='P-value', na_position='last')
print('-' * 100)
print(summary_df)
summary_df.to_csv("selected_feature_summary_2_4.csv", index=False)  # 保存したい場合
print('-' * 100)

# === 確認 ===
print("✅ selected_features特徴量:")
print(selected_features)

print('-' * 100)
########################################################################################
#Coefficients and Oddratio
########################################################################################
########################################################################################
from sklearn.linear_model import LogisticRegression

# データの準備
X = merged[selected_features]
y = merged['death']
from sklearn.impute import SimpleImputer

from sklearn.impute import SimpleImputer


print('-' * 100)

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
imputer = SimpleImputer(strategy='mean')  # または 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
# モデル学習
clf = LogisticRegression(max_iter=10000)
clf.fit(X_scaled, y)

# 係数とオッズ比
odds_ratios = np.exp(clf.coef_[0])
feature_odds = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': clf.coef_[0],
    'Odds Ratio': odds_ratios
})
print(feature_odds.sort_values('Odds Ratio', ascending=False))

print('-' * 100)

########################################################################################
#20 times 5-fold cross-validation Logistic Regression
########################################################################################
########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, roc_auc_score
from collections import defaultdict

# === ここで merged と selected_features を用意 ===
# 例:
# merged = pd.read_csv("your_data.csv")
# selected_features = [...]

X = merged[selected_features].dropna()
y = merged.loc[X.index, 'death'].astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
#param_grid = {'C': [0.001,0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2'], 'solver': ['newton-cg']}
grid = ParameterGrid(param_grid)
n_runs = 20
n_splits = 5
threshold_range = np.arange(0.01, 0.5, 0.001)
mean_fpr = np.linspace(0, 1, 100)
tprs_runs = []
all_aucs = []

# === スレッショルド別全ラン記録用 ===
threshold_metrics_all_runs = defaultdict(list)
from tqdm import tqdm

for run in tqdm(range(n_runs), desc="Logistic Regression Runs"):
#for run in range(n_runs):
    seed = 100 + run
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params = None

    # === ハイパーパラメータチューニング（F1） ===
    for params in grid:
        f1s = []
        for train_idx, test_idx in cv.split(X_scaled, y):
            model = LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000, **params)
            model.fit(X_scaled[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X_scaled[test_idx])
            f1s.append(f1_score(y.iloc[test_idx], y_pred))
        mean_f1 = np.mean(f1s)
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    # === ベストパラメータで再CV ===
    all_probs, all_trues = [], []
    tprs, aucs = [], []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 100)

    for train_idx, test_idx in cv.split(X_scaled, y):
        model = LogisticRegression(C=best_params['C'], penalty='l2', solver='lbfgs',
                                   class_weight='balanced', random_state=seed+100, max_iter=1000)
        model.fit(X_scaled[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X_scaled[test_idx])[:, 1]
        y_true = y.iloc[test_idx].values

        all_probs.extend(y_prob)
        all_trues.extend(y_true)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    all_probs = np.array(all_probs)
    all_trues = np.array(all_trues)
    all_aucs.append(np.mean(aucs))
    tprs_runs.append(np.mean(tprs, axis=0))

    # === 各スレッショルドごとの性能評価 ===
    roc_auc = roc_auc_score(all_trues, all_probs)
    for threshold in threshold_range:
        y_pred = (all_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_trues, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(all_trues, y_pred)
        f1 = f1_score(all_trues, y_pred)
        precision = precision_score(all_trues, y_pred, zero_division=0)
        threshold_metrics_all_runs[threshold].append({
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'roc_auc': roc_auc
        })


# === ROC カーブも描画 ===
plt.figure(figsize=(8, 6))
mean_tpr = np.mean(tprs_runs, axis=0)
std_tpr = np.std(tprs_runs, axis=0)
mean_auc = np.mean(all_aucs)
std_auc = np.std(all_aucs)

# === 平均・標準偏差をまとめる ===
avg_results = []
for threshold, metrics_list in threshold_metrics_all_runs.items():
    result = {'threshold': threshold}
    for metric in ['sensitivity', 'specificity', 'accuracy', 'f1', 'precision']:
        vals = [m[metric] for m in metrics_list]
        result[f'{metric}'] = np.mean(vals)
        #result[f'{metric}_std'] = np.std(vals)
    result['roc_auc'] = mean_auc
    avg_results.append(result)





plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})', lw=2)
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='b', alpha=0.2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC over 20 Runs (5-Fold CV)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === 保存 ===
df_avg = pd.DataFrame(avg_results)
df_avg = df_avg.sort_values(by="threshold").round(4)
df_avg.to_csv("all_threshold_metrics_0.001_2_4.csv", index=False, encoding='utf-8-sig')



