import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from glob import glob

# データを読み込む関数
def load_data(file_path):
    return np.load(file_path)

# リサンプリングを行う関数
def resample_data(data, original_rate, new_rate):
    duration = data.shape[1] / original_rate
    new_length = int(duration * new_rate)
    resampled_data = mne.filter.resample(data, up=new_rate, down=original_rate)
    return resampled_data

# アーチファクト除去を行う関数
def remove_artifacts(data, n_components=20):
    info = mne.create_info(ch_names=['eeg']*data.shape[0], sfreq=200, ch_types=['eeg']*data.shape[0])
    raw = mne.io.RawArray(data, info)
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
    ica.fit(raw)
    raw_ica = ica.apply(raw)
    return raw_ica.get_data()

# 正規化を行う関数
def normalize_data(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data

# 前処理を統合する関数
def preprocess_data(file_path):
    data = load_data(file_path)
    
    # データを float64 型に変換
    data = data.astype(np.float64)
    
    # リサンプリング
    data = resample_data(data, original_rate=200, new_rate=100)
    
    # アーチファクト除去
#     data = remove_artifacts(data)
    
    # 正規化
    data = normalize_data(data)
    
    return data


label = ["train","val","test"]
# label = ["val","test"]
for split in label:
    data_dir = 'data/'+split+'_X'
    print(len(glob(os.path.join(data_dir, "*.npy"))))

for split in label:
    # 保存先ディレクトリを指定
    data_dir = 'data/'+split+'_X'
    save_dir = 'data/filtered/'+split+'_X'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(glob(os.path.join(data_dir, "*.npy")))):
        # データの読み込み
        file_name = str(i).zfill(5) + ".npy"
        data_path = os.path.join(data_dir, file_name)
        preprocessed_data = preprocess_data(data_path)
        
        # フィルタリング後のデータを保存
        filtered_filename = os.path.join(save_dir, file_name)
        np.save(filtered_filename, preprocess_data)
        
        if i % 1000 == 0:
            print(i)
    
    print(save_dir)


# import shutil
# shutil.make_archive('archive_data', format='zip', root_dir='data/filtered')