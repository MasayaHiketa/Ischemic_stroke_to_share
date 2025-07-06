import pandas as pd

# 対象のICD9コード
icd_codes = ['43301', '43311', '43321', '43331', '43381', '43391', '43401', '43411', '43491']

# ICD9コードの列名一覧
icd_cols = [f'icd9_code{str(i).zfill(2)}' for i in range(1, 40)]

# BASICファイルから読み込み（ICD9コード列は文字列として）
df = pd.read_csv('20241206_010627mimic-iii-wave-match-BasicTable-icd.csv', dtype={col: str for col in icd_cols})

# フィルタ1: testDuring が "withinICU"
df = df[df['testDuring'] == 'withinICU']

# フィルタ2: ICD9コードのいずれかの列に対象コードが1つでも含まれている
icd_match = df[icd_cols].isin(icd_codes).any(axis=1)
df = df[icd_match]

# フィルタ3: sig_name に 'II' を含む（部分一致）
df = df[df['sig_name'].str.contains('II', na=False)]

# 各ユニーク患者の最終行だけを抽出
#df_last_rows = df.groupby('subject_id', as_index=False).tail(1)
df_last_rows = df
# 保存
df_last_rows.to_csv('filtered.csv', index=False)

# 出力ログ
print(f"抽出後の総行数（ユニーク患者数）: {len(df_last_rows)}")

###################################################################################################
###################################################################################################
###################################################################################################
import pandas as pd
from datetime import timedelta

# データ読み込み（時間系カラムをパース）
df = pd.read_csv('filtered.csv', parse_dates=['outtime', 'deathtime'])



# 結果を蓄積するリスト
result_rows = []

# ユニークな subject_id ごとに処理
for subject_id, group in df.groupby('subject_id'):
    # death のフラグ（全行同じと仮定）
    death_flag = group['death'].iloc[0]
    
    if death_flag == 0:
        # 生存者：最後の outtime を取得
        last_outtime = group['outtime'].max()
        target_time = last_outtime
    else:
        # 死亡者：deathtime を取得（1つだけあると仮定）
        target_time = group['deathtime'].iloc[0]
    
    # 2, 4, 6, 8, 10, 12時間前を計算
    time_minus_2h = target_time - timedelta(hours=2)
    time_minus_4h = target_time - timedelta(hours=4)
    time_minus_6h = target_time - timedelta(hours=6)
    time_minus_8h = target_time - timedelta(hours=8)
    time_minus_10h = target_time - timedelta(hours=10)
    time_minus_12h = target_time - timedelta(hours=12)

    # 結果に追加
    result_rows.append({
        'subject_id': subject_id,
        'death_flag': death_flag,
        'target_time': target_time,
        'time_minus_2h': time_minus_2h,
        'time_minus_4h': time_minus_4h,
        'time_minus_6h': time_minus_6h,
        'time_minus_8h': time_minus_8h,
        'time_minus_10h': time_minus_10h,
        'time_minus_12h': time_minus_12h,
    })

# DataFrame に変換
result_df = pd.DataFrame(result_rows)

# 保存
result_df.to_csv('time.csv', index=False)

#print("✅ ユニーク患者ごとの時間表を保存しました：death_or_outtime_summary.csv")


###################################################################################################
###################################################################################################
###################################################################################################

import pandas as pd
from datetime import timedelta

# データ読み込み
scan_df = pd.read_csv('20241207_174624mimic-iii-wave-match-DataScan.csv', parse_dates=['ecgDate'])
time_df = pd.read_csv('time.csv', parse_dates=['time_minus_6h', 'time_minus_8h'])

# ECG終了時間（ecgEndTime）を計算
scan_df['ecgEndTime'] = scan_df['ecgDate'] + pd.to_timedelta(scan_df['min'], unit='m')

# time_df を subject_id をキーにマージ
merged_df = pd.merge(scan_df, time_df, on='subject_id', how='inner')

# 開始と終了の重なり時間を計算
merged_df['overlap_start'] = merged_df[['ecgDate', 'time_minus_8h']].max(axis=1)
merged_df['overlap_end'] = merged_df[['ecgEndTime', 'time_minus_6h']].min(axis=1)

# オーバーラップの長さ（分）を計算
merged_df['overlap_duration'] = (merged_df['overlap_end'] - merged_df['overlap_start']).dt.total_seconds() / 60

# 条件: オーバーラップ時間が90分以上かつ、重なっていること
condition = merged_df['overlap_duration'] >= 0

# 条件に合う行を抽出し subject_id で並べ替え
filtered_df = merged_df[condition].sort_values('subject_id')

# 保存
filtered_df.to_csv('select.csv', index=False)

print(f"✅ 抽出された行数: {len(filtered_df)}（該当患者のみ、1.5時間以上重なりあり）")
