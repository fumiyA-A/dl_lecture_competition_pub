import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
from glob import glob


# サンプリングレート
fs = 200  # Hz

# ローパスフィルタの設計
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # ナイキスト周波数
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    y = y.astype(np.float32)
    return y

# ローパスフィルタの適用
cutoff = 50  # カットオフ周波数を50Hzに設定

label = ["train", "val", "test"]
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
        data = np.load(data_path)

        filtered_data = lowpass_filter(data, cutoff, fs)

        # フィルタリング後のデータを保存
        filtered_filename = os.path.join(save_dir, file_name)
        np.save(filtered_filename, filtered_data)
    
    
