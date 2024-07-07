import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os

# 保存先ディレクトリを指定
save_dir = '/data_filtered'
load_dir = '/train'
os.makedirs(save_dir, exist_ok=True)

file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
for i in range(len())
X_path = os.path.join('/data', f"{load_dir}_X", str(i).zfill(5) + ".npy")
X = torch.from_numpy(np.load(X_path))
# データの読み込み
data = np.load(filename)

# サンプリングレート
fs = 200  # Hz
t = np.arange(data.shape[0]) / fs  # 時間軸

# ローパスフィルタの設計
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # ナイキスト周波数
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

# ローパスフィルタの適用
cutoff = 50  # カットオフ周波数を50Hzに設定
filtered_data = lowpass_filter(data, cutoff, fs)

# フィルタリング後のデータを保存
filtered_filename = os.path.join(save_dir, 'filtered_00000.npy')
np.save(filtered_filename, filtered_data)

# 全チャンネルのフィルタリング後のデータを時系列グラフとして表示
plt.figure(figsize=(15, 10))
for i in range(data.shape[1]):
    plt.plot(t, filtered_data[:, i], label=f'Channel {i+1}')

plt.title('Filtered Time Series Data for All Channels')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.grid()
plt.show()

# 特定のチャンネル（例: チャンネル0）のフィルタリング後のデータを時系列グラフとして表示
channel_index = 0  # 表示したいチャンネルのインデックス
plt.figure(figsize=(15, 5))
plt.plot(t, filtered_data[:, channel_index])
plt.title(f'Filtered Time Series Data for Channel {channel_index + 1}')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
