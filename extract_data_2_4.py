import pandas as pd
import numpy as np
import wfdb
import os
from datetime import timedelta
from scipy.signal import butter, filtfilt, iirnotch  
from wfdb import processing
from scipy.signal import butter, filtfilt, iirnotch
import neurokit2 as nk
import matplotlib.pyplot as plt
from kde import *
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import warnings

# ignore ALL RuntimeWarnings (you can tighten this if you know the message text)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# or even more specifically, match the message:
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*invalid value encountered in divide.*"
)
# - preprocess_ecg
def plot_ecg_signal(time, signal):
    plt.cla()
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(100, 5))
    ax = plt.axes()
    ax.plot(time, signal)
    # setup major and minor ticks
    min_t = int(np.min(time))
    max_t = round(np.max(time))
    major_ticks = np.arange(min_t, max_t+1)
    ax.set_xticks(major_ticks)
    # Turn on the minor ticks on
    ax.minorticks_on()
    # Make the major grid
    ax.grid(which='major', linestyle='-', color='red', linewidth='1.0')
    # Make the minor grid
    ax.grid(which='minor', linestyle=':', color='black', linewidth='0.5')
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    return ax
# https://www.indigits.com/post/2022/10/ecg_python/
# - processSubEcg
def getECGfeatures(tECG_R_Peaks, fss):
    out = pd.DataFrame()
    rrPoincart = pd.DataFrame()
    try:
        out = nk.hrv(tECG_R_Peaks, sampling_rate=fss)
        out = out.dropna(axis=1).reset_index(drop=True)
        out = out[np.isfinite(out).all(1)]
    except:
        print('getECGfeatures error: nk.hrv not work')
        return pd.DataFrame(), pd.DataFrame()
    try:
        rrIntv = processing.calc_rr(tECG_R_Peaks, fs=fss)
        rrIntv = np.array(rrIntv)
        rrIntv = rrIntv/np.int64(np.ceil(fss*1.0))
        rrPoincart = pd.DataFrame()
        rrPoincart['rrn'] = rrIntv[:-1]
        rrPoincart['rrn+1'] = rrIntv[1:]
        out['rrIntv_max'] = np.max(rrIntv)
        out['rrIntv_min'] = np.min(rrIntv)
        out['rrIntv_mean'] = np.mean(rrIntv)
        out['rrIntv_std'] = np.std(rrIntv)
        out['rrIntv_quantile25'] = np.quantile(rrIntv, 0.25)
        out['rrIntv_quantile50'] = np.quantile(rrIntv, 0.5)
        out['rrIntv_quantile75'] = np.quantile(rrIntv, 0.75)
        out['rrIntv_percentile25'] = np.percentile(rrIntv, 25)
        out['rrIntv_percentile50'] = np.percentile(rrIntv, 50)
        out['rrIntv_percentile75'] = np.percentile(rrIntv, 75)
        zt = pd.DataFrame()
        zt['rrn'] = rrIntv
        out['rrIntv_MaxHistCounts'] = zt['rrn'].value_counts().values[0]
    except:
        print('getECGfeatures error: processing.calc_rr not work')
        return out, pd.DataFrame()
    try:
        values = np.vstack(rrPoincart[['rrn', 'rrn+1']].to_numpy())
        erakde_params = [
            (3.0, 0.6),
            (1.5, 0.5),
            (1.5, 0.3),
            (1.2, 0.5),
            (1.2, 0.3),
            (1.0, 0.5),
            (1.0, 0.3)
        ]

        for beta, factor in erakde_params:
            label = f"erakde_b{beta}_f{factor}"

            erakde = JitERAKDE(values, len(values)-1, beta, factor)
            eva = erakde.get_densities(values)

            out[f"{label}_max"] = np.max(eva)
            out[f"{label}_min"] = np.min(eva)
            out[f"{label}_mean"] = np.mean(eva)
            out[f"{label}_std"] = np.std(eva)
            out[f"{label}_median"] = np.median(eva)

            out[f"{label}_quantile25"] = np.quantile(eva, 0.25)
            out[f"{label}_quantile50"] = np.quantile(eva, 0.5)
            out[f"{label}_quantile75"] = np.quantile(eva, 0.75)

            out[f"{label}_percentile25"] = np.percentile(eva, 25)
            out[f"{label}_percentile50"] = np.percentile(eva, 50)
            out[f"{label}_percentile75"] = np.percentile(eva, 75)

            out[f"{label}_decile10"] = np.percentile(eva, 10)
            out[f"{label}_decile90"] = np.percentile(eva, 90)

    except:
        print('getECGfeatures error: JitERAKDE')

    return out, rrPoincart

def processSubEcg(subEcgSignal, fss):
    try:
        print(f"nk.ecg_process 開始: 信号長 = {len(subEcgSignal)}, Fs = {fss}")
        preCleanProSignal, info = nk.ecg_process(subEcgSignal, sampling_rate=fss)
    except Exception as e:
        print(f"processSubEcg ecg_process error: {type(e).__name__}: {e}")
        return []

    try:
        print("nk.ecg_quality 実行中...")
        qualitysss = nk.ecg_quality(
            preCleanProSignal['ECG_Clean'],
            rpeaks=info['ECG_R_Peaks'],
            method='zhao2018',
            approach='fuzzy',
            sampling_rate=int(fss)
        )
        print(f"信号品質評価: {qualitysss}")
        if qualitysss == 'Unacceptable':
            return []

        if len(info['ECG_R_Peaks']) > 0:
            return info['ECG_R_Peaks']
        else:
            return []
    except Exception as e:
        print(f"processSubEcg ecg_quality error: {type(e).__name__}: {e}")
        return []

    return []

# https://neuropsychology.github.io/NeuroKit/functions/ecg.html#neurokit2.ecg.ecg_quality
# - getECGfeatures
# Bandpass filter for ECG (0.5-50 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
# Notch filter to remove 50/60 Hz powerline interference
def notch_filter(data, notch_freq, fs, quality_factor=30):
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data)
# Denoising process
def preprocess_ecg(ecg_signal, fs):
    # Step 1: Bandpass Filter
    ecg_bandpassed = bandpass_filter(ecg_signal, 0.5, 50, fs)

    # Step 2: Notch Filter
    ecg_notched = notch_filter(ecg_bandpassed, 50, fs)

    # Step 3: Moving Average (Optional)
    ecg_smoothed = np.convolve(ecg_notched, np.ones(3)/3, mode='same')

    return ecg_smoothed

# select.csv だけ読み込む（time.csvは使わない）
merged_df = pd.read_csv("select.csv", dtype={'subject_id': int}, parse_dates=['time_minus_2h', 'time_minus_4h'])

# 確認出力
print(f"select.csv にある患者数: {merged_df['subject_id'].nunique()}")
print(merged_df.columns)

import platform
import os

import platform

def normalize_load_path(path):
    os_type = platform.system()

    if os_type == 'Windows':
        replace_base = r"\\140.112.28.172\mimic3wdb-matched-v1.0\\"
    elif os_type == 'Darwin':  # macOS
        replace_base = "/Volumes/mimic3wdb-matched-v1.0/"
    elif os_type == 'Linux':
        replace_base = "/mnt/mimic3wdb-matched-v1.0/"
    else:
        raise OSError("Unsupported OS")

    # 置換
    path = path.replace("M:\\", replace_base)
    path = path.replace("M:/", replace_base)

    # Windows以外ならバックスラッシュをスラッシュに統一
    if os_type != 'Windows':
        path = path.replace("\\", "/")

    return path

# 適用
merged_df['loadPath'] = merged_df['loadPath'].apply(normalize_load_path)


all_hrv = []
import os
import shutil  # これを追加
# ローカルにコピーするフォルダを準備
local_dir = './tmp_ecg'
os.makedirs(local_dir, exist_ok=True)
import re
from datetime import datetime, timedelta
# ここで merged_df が既に読み込まれていると仮定します

# === 設定 ===
fs = 125  # サンプリング周波数
window_size_min = 5
window_size = fs * 60 * window_size_min  # サンプル数（10分）

max_windows = 24  # 最大で12区間まで
all_hrv = []
local_dir = "./temp_segments"

os.makedirs(local_dir, exist_ok=True)

label_df = pd.read_csv(r"20241206_010627mimic-iii-wave-match-BasicTable-icd.csv")
dead_ids = label_df[label_df['death'] == 1]['subject_id'].unique()

hrv_path = 'extracted_data_2_4_for_you.csv'

# 既存ファイルの subject_id を読み込む（あれば）
existing_ids = set()
if os.path.exists(hrv_path):
    existing_df = pd.read_csv(hrv_path)
    existing_ids = set(existing_df['subject_id'].unique())

# === データ処理ループ ===
for idx, row in merged_df.iterrows():
    subject_id = row['subject_id']
    hea_path = row['loadPath']
    ecg_datetime = row['ecgDatetime']
    t_start = row['time_minus_4h']
    t_end = row['time_minus_2h']

    if subject_id in existing_ids:
        print(f"Skip: subject_id {subject_id} は既に存在します")
        continue
    # ✅ 死亡者ならスキップ
    #if subject_id not in dead_ids:
        #print(f"☠️ Skip (death=1): subject_id {subject_id} は死亡者でない")
        #continue

    if not hea_path.endswith('.hea'):
        hea_path += '.hea'

    try:
        if not os.path.exists(hea_path):
            print(f"❌ .hea ファイルが見つかりません: {hea_path}")
            continue

        with open(hea_path, 'r') as f:
            lines = f.readlines()

        header_line = lines[0].strip()
        header_parts = header_line.split()
        base_time_str = header_parts[4]
        base_date_str = header_parts[5]
        ecg_start_time = pd.to_datetime(base_date_str + ' ' + base_time_str, format="%d/%m/%Y %H:%M:%S.%f")

        current_time = ecg_start_time
        segments = []

        for line in lines[1:]:
            if line.startswith("~"):
                try:
                    skip_samples = int(line.strip().split()[1])
                    skip_seconds = skip_samples / fs
                    current_time += pd.to_timedelta(skip_seconds, unit='s')
                except:
                    continue
                continue

            parts = line.strip().split()
            if len(parts) < 2 or not re.match(r"^\d+_\d+$", parts[0]):
                continue

            seg_name = parts[0]
            seg_samples = int(parts[1])
            seg_start = current_time
            seg_end = current_time + pd.to_timedelta(seg_samples / fs, unit='s')

            if seg_end >= t_start and seg_start <= t_end:
                overlap_start = max(seg_start, t_start)
                overlap_end = min(seg_end, t_end)
                overlap_samples = int((overlap_end - overlap_start).total_seconds() * fs)

                if overlap_samples >= window_size:
                    segments.append({
                        'segment': seg_name,
                        'start': seg_start,
                        'end': seg_end
                    })

            current_time = seg_end

        if not segments:
            print("❌ 対象範囲に一致するセグメントなし")
            continue

        collected = 0

        for s in segments:
            if collected >= max_windows:
                break

            seg_name = s['segment']
            seg_start = s['start']
            seg_end = s['end']

            overlap_start = max(seg_start, t_start)
            overlap_end = min(seg_end, t_end)

            fs_start = int((overlap_start - seg_start).total_seconds() * fs)
            fs_end = int((overlap_end - seg_start).total_seconds() * fs)

            # ファイルコピー
            full_base = os.path.join(os.path.dirname(hea_path), seg_name)
            local_hea = os.path.join(local_dir, seg_name + '.hea')
            local_dat = os.path.join(local_dir, seg_name + '.dat')

            shutil.copy(full_base + '.hea', local_hea)
            shutil.copy(full_base + '.dat', local_dat)

            try:
                record = wfdb.rdrecord(os.path.join(local_dir, seg_name))

                if 'II' in record.sig_name:
                    ecg_raw = record.p_signal[:, record.sig_name.index('II')]
                else:
                    print(f"⚠️ II誘導が見つかりません: {seg_name}")
                    continue

                segment_signal = ecg_raw[fs_start:fs_end]
                segment_signal = pd.Series(segment_signal).dropna().to_numpy()

                # 🧠 セグメントの中から複数の10分区間を取り出す
                num_possible = len(segment_signal) // window_size

                for i in range(num_possible):
                    if collected >= max_windows:
                        break

                    start = i * window_size
                    end = start + window_size
                    sub_signal = segment_signal[start:end]

                    clean_signal = preprocess_ecg(sub_signal, fs)
                    r_peaks = processSubEcg(clean_signal, fs)

                    if len(r_peaks) == 0:
                        continue

                    hrv_feat, _ = getECGfeatures(r_peaks, fs)
                    if hrv_feat.empty:
                        continue

                    # 追加情報をDataFrameとしてまとめる
                    metadata = pd.DataFrame({
                        'subject_id': [subject_id],
                        'segment': [seg_name],
                        'window_index': [collected],
                        'window_size': [window_size]
                    })

                    # hrv_feat が行1つのDataFrameである前提（多くのHRVツールはそう）
                    hrv_feat = pd.concat([metadata.reset_index(drop=True), hrv_feat.reset_index(drop=True)], axis=1)

                    # 最後にリストへ追加
                    all_hrv.append(hrv_feat)

                    collected += 1

            except Exception as e:
                print(f"❌ 読み込み失敗: {type(e).__name__}: {e}")
                continue

    except Exception as e:
        print(f"❌ 全体エラー: {type(e).__name__}: {e}")
        continue

# 保存処理
if all_hrv:
    new_df = pd.concat(all_hrv, ignore_index=True)

    if os.path.exists(hrv_path):
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_csv(hrv_path, index=False)
    print(f"✅ 保存完了: {hrv_path}")
else:
    print("有効なHRVデータはありませんでした。")
