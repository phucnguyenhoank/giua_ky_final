# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import re
import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# Try to import tensorflow_addons, if not available, define a fallback
try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    print("TensorFlow Addons not found. Using standard TensorFlow components instead.")
    # Install TensorFlow Addons if running in Colab
    try:
        import google.colab
        print("Installing TensorFlow Addons...")
        !pip install -q tensorflow-addons
        import tensorflow_addons as tfa
        HAS_TFA = True
    except (ImportError, ModuleNotFoundError):
        print("Not running in Colab or couldn't install TensorFlow Addons.")
        HAS_TFA = False

# Thiết lập seed cho tính tái tạo
np.random.seed(42)
tf.random.set_seed(42)

# Kiểm tra GPU
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Đặt bộ nhớ GPU động để tránh chiếm toàn bộ bộ nhớ GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")

# Kết nối Google Drive (nếu chạy trong Colab)
try:
    from google.colab import drive
    print("Kết nối Google Drive...")
    drive.mount('/content/drive')
except:
    print("Không thể kết nối Google Drive hoặc không chạy trong môi trường Colab.")

# Đường dẫn đến thư mục chứa dữ liệu BIDMC
BIDMC_DATA_DIR = '/content/drive/MyDrive/codePPG2025/bidmc/bidmc_csv'
WORK_DIR = os.getcwd()
PROCESSED_DIR = os.path.join(WORK_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Hàm tổ chức file BIDMC
def organize_bidmc_files(files):
    """
    Phân loại các file BIDMC thành signals, numerics, và breaths
    với hỗ trợ nhiều định dạng tên file khác nhau
    """
    organized = {}

    for file in files:
        basename = os.path.basename(file)

        # Thử các mẫu tên file khác nhau
        match = None

        # Mẫu: bidmc_XX_Type.csv
        pattern1 = re.search(r'bidmc_(\d+)_([A-Za-z]+)\.csv', basename)
        if pattern1:
            patient_id = pattern1.group(1)
            file_type = pattern1.group(2).lower()
            match = (patient_id, file_type)

        # Mẫu: bidmcXX_Type.csv
        if not match:
            pattern2 = re.search(r'bidmc(\d+)_([A-Za-z]+)\.csv', basename)
            if pattern2:
                patient_id = pattern2.group(1)
                file_type = pattern2.group(2).lower()
                match = (patient_id, file_type)

        # Mẫu: bidmc-XX-Type.csv
        if not match:
            pattern3 = re.search(r'bidmc-(\d+)-([A-Za-z]+)\.csv', basename)
            if pattern3:
                patient_id = pattern3.group(1)
                file_type = pattern3.group(2).lower()
                match = (patient_id, file_type)

        # Mẫu: bidmcXXType.csv
        if not match:
            pattern4 = re.search(r'bidmc(\d+)([A-Za-z]+)\.csv', basename)
            if pattern4:
                patient_id = pattern4.group(1)
                file_type = pattern4.group(2).lower()
                match = (patient_id, file_type)

        if match:
            patient_id, file_type = match

            # Thêm bệnh nhân vào từ điển nếu chưa có
            if patient_id not in organized:
                organized[patient_id] = {'signals': None, 'numerics': None, 'breaths': None}

            # Phân loại loại file
            if 'signal' in file_type.lower():
                organized[patient_id]['signals'] = file
            elif 'numeric' in file_type.lower():
                organized[patient_id]['numerics'] = file
            elif 'breath' in file_type.lower():
                organized[patient_id]['breaths'] = file

    # Kiểm tra và chỉ giữ lại các bệnh nhân có ít nhất file signals và numerics
    complete_patients = {}
    for patient_id, files in organized.items():
        if files['signals'] and files['numerics']:
            complete_patients[patient_id] = files

    return complete_patients

# Tìm tất cả các file CSV
print(f"Đang tìm các file CSV trong thư mục: {BIDMC_DATA_DIR}")
bidmc_files = glob.glob(os.path.join(BIDMC_DATA_DIR, '*.csv'))
print(f"Tìm thấy {len(bidmc_files)} file CSV")

# Tổ chức các file
patient_files = organize_bidmc_files(bidmc_files)
print(f"\nTìm thấy {len(patient_files)} bệnh nhân có đủ dữ liệu (signals + numerics)")

# Hiển thị thông tin cho bệnh nhân đầu tiên (nếu có)
if patient_files:
    first_patient = list(patient_files.keys())[0]
    print(f"\nFiles cho bệnh nhân {first_patient}:")
    for file_type, file_path in patient_files[first_patient].items():
        if file_path:
            print(f"{file_type}: {os.path.basename(file_path)}")

# Kiểm tra xem có đủ dữ liệu không
if len(patient_files) == 0:
    raise ValueError("Không tìm thấy bệnh nhân nào có đủ dữ liệu! Kiểm tra lại dữ liệu.")

# Kiểm tra cấu trúc file tín hiệu và thông số cho bệnh nhân đầu tiên
def check_file_structure(patient_id, patient_files):
    print(f"\nKiểm tra cấu trúc file cho bệnh nhân {patient_id}")

    # Kiểm tra file tín hiệu
    signal_file = patient_files[patient_id]['signals']
    if signal_file:
        signals_df = pd.read_csv(signal_file)
        print("Signal file columns:", signals_df.columns.tolist())
        print("Signal file shape:", signals_df.shape)
        print("Signal file sample:", signals_df.head(3))

    # Kiểm tra file thông số
    numeric_file = patient_files[patient_id]['numerics']
    if numeric_file:
        numerics_df = pd.read_csv(numeric_file)
        print("\nNumeric file columns:", numerics_df.columns.tolist())
        print("Numeric file shape:", numerics_df.shape)
        print("Numeric file sample:", numerics_df.head(3))

check_file_structure(first_patient, patient_files)

# Hàm xử lý dữ liệu
def process_patient_data(patient_id, patient_files):
    """
    Xử lý dữ liệu của một bệnh nhân
    """
    print(f"\nĐang xử lý dữ liệu bệnh nhân {patient_id}...")

    # Tải dữ liệu
    signals_df = pd.read_csv(patient_files[patient_id]['signals'])
    numerics_df = pd.read_csv(patient_files[patient_id]['numerics'])

    # Xác định cột PPG, HR và RESP dựa trên tên cột
    ppg_cols = [col for col in signals_df.columns if 'pleth' in col.lower()]
    hr_cols = [col for col in numerics_df.columns if 'hr' in col.lower()]
    br_cols = [col for col in numerics_df.columns if 'resp' in col.lower()]

    # Kiểm tra xem có tìm thấy cột phù hợp không
    if not ppg_cols:
        print(f"Không tìm thấy cột PPG trong file tín hiệu của bệnh nhân {patient_id}")
        # Thử tìm cột có chứa 'p' hoặc 'pleth'
        ppg_cols = [col for col in signals_df.columns if ' p' in col.lower()]
        if not ppg_cols:
            print("Các cột có sẵn:", signals_df.columns.tolist())
            return None

    if not hr_cols:
        print(f"Không tìm thấy cột HR trong file thông số của bệnh nhân {patient_id}")
        return None

    # Chọn cột đầu tiên tìm được
    ppg_col = ppg_cols[0]
    hr_col = hr_cols[0]
    br_col = br_cols[0] if br_cols else None

    print(f"Đang sử dụng cột: PPG={ppg_col}, HR={hr_col}")
    if br_col:
        print(f"Cột BR={br_col}")
    else:
        print("Không tìm thấy cột BR, sẽ sử dụng giá trị mặc định")

    # Trích xuất tín hiệu PPG
    ppg_signal = signals_df[ppg_col].values

    # Lọc nhiễu
    def apply_bandpass_filter(data, fs=125, lowcut=0.5, highcut=8.0):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)

    # Chuẩn hóa tín hiệu PPG
    def normalize_signal(data):
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    # Lọc và chuẩn hóa PPG
    ppg_filtered = apply_bandpass_filter(ppg_signal)
    ppg_normalized = normalize_signal(ppg_filtered)

    # Phân đoạn tín hiệu (cửa sổ 10 giây, 1250 mẫu)
    window_size = 10  # seconds
    fs = 125  # sampling frequency
    samples_per_window = int(fs * window_size)

    # Đảm bảo độ dài tín hiệu là bội số của kích thước cửa sổ
    signal_length = len(ppg_normalized)
    num_windows = signal_length // samples_per_window

    ppg_segments = []
    segment_times = []

    for i in range(num_windows):
        start_idx = i * samples_per_window
        end_idx = start_idx + samples_per_window

        ppg_segments.append(ppg_normalized[start_idx:end_idx])
        segment_times.append(i * window_size)

    # Trích xuất HR và BR cho mỗi phân đoạn
    hrs = []
    brs = []

    for time_start in segment_times:
        time_end = time_start + window_size

        # Tìm HR trong khoảng thời gian
        segment_hr = numerics_df[(numerics_df['Time [s]'] >= time_start) &
                                 (numerics_df['Time [s]'] < time_end)]

        if len(segment_hr) > 0:
            hr = segment_hr[hr_col].mean()
            br = segment_hr[br_col].mean() if br_col else np.nan
        else:
            # Nếu không có dữ liệu trong khoảng, lấy giá trị gần nhất
            closest_idx = (numerics_df['Time [s]'] - time_start).abs().idxmin()
            hr = numerics_df.loc[closest_idx, hr_col]
            br = numerics_df.loc[closest_idx, br_col] if br_col else np.nan

        # Nếu không có dữ liệu BR, dùng giá trị mặc định
        if np.isnan(br):
            br = 15.0  # Giá trị mặc định

        hrs.append(hr)
        brs.append(br)

    # Tạo kết quả
    return {
        'patient_id': patient_id,
        'ppg_segments': np.array(ppg_segments),
        'segment_times': np.array(segment_times),
        'hrs': np.array(hrs),
        'brs': np.array(brs)
    }

# Xử lý một bệnh nhân để kiểm tra
patient_data = process_patient_data(first_patient, patient_files)

if patient_data:
    print("\nKết quả xử lý:")
    print(f"Số lượng phân đoạn: {len(patient_data['ppg_segments'])}")
    print(f"Kích thước phân đoạn PPG: {patient_data['ppg_segments'].shape}")
    print(f"Phạm vi HR: {np.min(patient_data['hrs']):.1f} - {np.max(patient_data['hrs']):.1f} BPM")
    print(f"Phạm vi BR: {np.min(patient_data['brs']):.1f} - {np.max(patient_data['brs']):.1f} BrPM")

    # Vẽ một số phân đoạn
    plt.figure(figsize=(15, 5))

    for i in range(min(3, len(patient_data['ppg_segments']))):
        plt.subplot(1, 3, i + 1)
        plt.plot(patient_data['ppg_segments'][i])
        plt.title(f"PPG - Segment {i}\nHR: {patient_data['hrs'][i]:.1f}, BR: {patient_data['brs'][i]:.1f}")

    plt.tight_layout()
    plt.savefig('ppg_segments_sample.png')
    plt.close()

    # Lưu dữ liệu đã xử lý
    np.savez(os.path.join(PROCESSED_DIR, f'patient_{first_patient}_processed.npz'),
             ppg_segments=patient_data['ppg_segments'],
             hrs=patient_data['hrs'],
             brs=patient_data['brs'])

    print(f"\nĐã lưu dữ liệu đã xử lý cho bệnh nhân {first_patient}")
else:
    print("Không thể xử lý dữ liệu bệnh nhân")

# Định nghĩa lớp Dataset TensorFlow
class BIDMCDataset:
    def __init__(self, data):
        self.ppg_segments = data['ppg_segments'].astype(np.float32)
        self.hrs = data['hrs'].astype(np.float32)
        self.brs = data['brs'].astype(np.float32)

        print(f"Dataset: {len(self.ppg_segments)} phân đoạn")
        print(f"HR stats: mean={np.mean(self.hrs):.1f}, std={np.std(self.hrs):.1f}, min={np.min(self.hrs):.1f}, max={np.max(self.hrs):.1f}")
        print(f"BR stats: mean={np.mean(self.brs):.1f}, std={np.std(self.brs):.1f}, min={np.min(self.brs):.1f}, max={np.max(self.brs):.1f}")

        # In ra một số mẫu để kiểm tra
        print("\nMẫu dữ liệu từ dataset:")
        for i in range(min(5, len(self.ppg_segments))):
            print(f"Mẫu #{i+1}:")
            print(f"  PPG shape: {self.ppg_segments[i].shape}")
            print(f"  PPG sample (5 giá trị đầu): {self.ppg_segments[i][:5]}")
            print(f"  HR: {self.hrs[i]:.2f} BPM")
            print(f"  BR: {self.brs[i]:.2f} BrPM")

    def create_tf_dataset(self, batch_size=32, shuffle=True, cache=True, prefetch=True):
        # Xử lý giá trị NaN
        ppg_segments = np.nan_to_num(self.ppg_segments, nan=0.0)
        hrs = np.nan_to_num(self.hrs, nan=75.0).reshape(-1, 1)
        brs = np.nan_to_num(self.brs, nan=15.0).reshape(-1, 1)

        # Tạo dataset từ numpy array
        dataset = tf.data.Dataset.from_tensor_slices((
            ppg_segments,
            hrs,
            brs
        ))

        # Shuffle nếu cần
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(ppg_segments))

        # Tạo batch và tối ưu hóa đường dẫn dữ liệu
        dataset = dataset.batch(batch_size)

        if cache:
            dataset = dataset.cache()
        
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

# Sampling Layer cho VAE (reparameterization trick)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder Network với tối ưu hóa
# Encoder Network with more advanced Conv1D operations
class Encoder(layers.Layer):
    def __init__(self, latent_dim=128, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        
        # Improved 1D CNN layers with residual connections and dilation
        self.conv_block1 = self._create_conv_block(32, 9, 1, 2)  # Output: 625 samples
        self.conv_block2 = self._create_conv_block(64, 7, 2, 2)  # Output: 312 samples
        self.conv_block3 = self._create_conv_block(128, 5, 4, 2) # Output: 156 samples
        self.conv_block4 = self._create_conv_block(256, 3, 8, 2) # Output: 78 samples
        
        # More effective parameter processing network
        self.param_encoder = tf.keras.Sequential([
            layers.Dense(64, name="param_dense1"),
            layers.BatchNormalization(name="param_bn1"),
            layers.LeakyReLU(0.2, name="param_leaky1"),
            layers.Dense(128, name="param_dense2"),  
            layers.BatchNormalization(name="param_bn2"),
            layers.LeakyReLU(0.2, name="param_leaky2")
        ], name="param_encoder")
        
        # Improved feature fusion network with residual connections
        self.combined_fc1 = layers.Dense(512, name="combined_dense1")
        self.combined_bn1 = layers.BatchNormalization(name="combined_bn1")
        self.combined_act1 = layers.LeakyReLU(0.2, name="combined_leaky1")
        self.combined_drop1 = layers.Dropout(0.3, name="combined_dropout1")
        
        self.combined_fc2 = layers.Dense(512, name="combined_dense2")
        self.combined_bn2 = layers.BatchNormalization(name="combined_bn2")
        self.combined_act2 = layers.LeakyReLU(0.2, name="combined_leaky2")
        self.combined_drop2 = layers.Dropout(0.3, name="combined_dropout2")
        
        # Latent space projection
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling(name="sampling")
        
        # Flatten layer for the conv output
        self.flatten = layers.Flatten()
    
    def _create_conv_block(self, filters, kernel_size, dilation_rate=1, pool_size=2):
        """Create a convolutional block with residual connection and dilation"""
        return tf.keras.Sequential([
            # Main convolution path
            layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            # Optional residual connection for deeper networks
            layers.Conv1D(filters, 1, padding="same"),  # 1x1 conv to keep dimensions
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            # Pooling for downsampling
            layers.MaxPooling1D(pool_size),
            
            # Dropout for regularization
            layers.SpatialDropout1D(0.1)
        ])
    
    def call(self, inputs, training=False):
        ppg, hr, br = inputs
        
        # Reshape PPG for 1D convolution if needed
        if len(ppg.shape) == 2:
            ppg = tf.expand_dims(ppg, axis=-1)  # Add channel dimension [batch, 1250, 1]
        
        # Process through CNN blocks
        x = self.conv_block1(ppg, training=training)
        x = self.conv_block2(x, training=training)
        x = self.conv_block3(x, training=training)
        x = self.conv_block4(x, training=training)
        
        # Flatten convolutional output
        conv_features = self.flatten(x)
        
        # Process HR and BR parameters
        params = tf.concat([hr, br], axis=1)
        param_features = self.param_encoder(params, training=training)
        
        # Combined feature processing with residual connection
        combined = tf.concat([conv_features, param_features], axis=1)
        
        # First dense block
        hidden = self.combined_fc1(combined)
        hidden = self.combined_bn1(hidden, training=training)
        hidden = self.combined_act1(hidden)
        hidden = self.combined_drop1(hidden, training=training)
        
        # Second dense block with residual connection
        residual = hidden
        hidden = self.combined_fc2(hidden)
        hidden = self.combined_bn2(hidden, training=training)
        hidden = self.combined_act2(hidden)
        hidden = self.combined_drop2(hidden, training=training)
        hidden = hidden + residual  # Residual connection
        
        # Generate latent variables
        z_mean = self.z_mean(hidden)
        z_log_var = self.z_log_var(hidden)
        z = self.sampling([z_mean, z_log_var])
        
        return z_mean, z_log_var, z

# Decoder Network với tối ưu hóa
# Decoder Network simplified for better shape handling
class Decoder(layers.Layer):
    def __init__(self, output_dim=1250, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        
        # Parameter processing
        self.param_encoder = tf.keras.Sequential([
            layers.Dense(64, name="param_dense1"),
            layers.BatchNormalization(name="param_bn1"),
            layers.LeakyReLU(0.2, name="param_leaky1"),
            layers.Dense(128, name="param_dense2"),  
            layers.BatchNormalization(name="param_bn2"),
            layers.LeakyReLU(0.2, name="param_leaky2")
        ], name="param_processor")
        
        # Feature fusion network
        self.combined_fc = tf.keras.Sequential([
            layers.Dense(512, name="combined_dense1"),
            layers.BatchNormalization(name="combined_bn1"),
            layers.LeakyReLU(0.2, name="combined_leaky1"),
            layers.Dropout(0.3, name="combined_dropout1"),
            layers.Dense(512, name="combined_dense2"),
            layers.BatchNormalization(name="combined_bn2"),
            layers.LeakyReLU(0.2, name="combined_leaky2"),
            layers.Dropout(0.3, name="combined_dropout2")
        ], name="combined_network")
        
        # Initial reshape to 1D sequence
        # To reach 1250 output length, we'll use a base length of 78 and 4 upsampling layers (78 * 2^4 = 1248)
        # We'll handle the final length adjustment at the end
        self.base_length = 78
        self.fc_to_seq = layers.Dense(self.base_length * 16, name="fc_to_seq")
        self.reshape = layers.Reshape((self.base_length, 16), name="reshape")
        
        # Transposed Convolution layers with fixed architecture
        self.conv_block1 = self._create_conv_block(256, 5, 2)  # 78 -> 156
        self.conv_block2 = self._create_conv_block(128, 5, 2)  # 156 -> 312
        self.conv_block3 = self._create_conv_block(64, 5, 2)   # 312 -> 624
        self.conv_block4 = self._create_conv_block(32, 5, 2)   # 624 -> 1248
        
        # Final output conversion
        self.final_conv = layers.Conv1D(1, 5, padding="same", name="final_conv")
        self.final_activation = layers.Activation("sigmoid", name="final_activation")
        
        # Fixed output adjustment layer - handles exact sizing to 1250
        self.output_adjustment = layers.Dense(output_dim, name="output_adjustment")
        
    def _create_conv_block(self, filters, kernel_size, upsample=1):
        """Create a convolutional block with upsampling"""
        return tf.keras.Sequential([
            # Upsampling using repeat-based approach
            layers.UpSampling1D(upsample),
            # Feature transformation
            layers.Conv1D(filters, kernel_size, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
    
    def call(self, inputs, training=False):
        z, hr, br = inputs
        
        # Process HR and BR parameters
        params = tf.concat([hr, br], axis=1)
        param_features = self.param_encoder(params, training=training)
        
        # Combine latent vector with parameters
        combined = tf.concat([z, param_features], axis=1)
        hidden = self.combined_fc(combined, training=training)
        
        # Convert to initial sequence representation
        seq_features = self.fc_to_seq(hidden)
        x = self.reshape(seq_features)
        
        # Apply transposed convolution layers
        x = self.conv_block1(x, training=training)
        x = self.conv_block2(x, training=training)
        x = self.conv_block3(x, training=training)
        x = self.conv_block4(x, training=training)
        
        # Apply final convolution and activation
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        # Squeeze the channel dimension to get [batch, output_length]
        x = tf.squeeze(x, axis=-1)
        
        # Use a final dense layer to force exact output dimension
        # This ensures we always get exactly self.output_dim values
        output = self.output_adjustment(x)
            
        return output

# Định nghĩa model cVAE với bổ sung nhiều tính năng
# Định nghĩa model cVAE với bổ sung nhiều tính năng
class CVAE(keras.Model):
    def __init__(self, 
                 latent_dim=128, 
                 input_dim=1250,
                 beta=1.0,
                 kl_weight=1.0,
                 recon_weight=1.0,
                 perceptual_weight=0.0,
                 spectral_weight=0.1,  # New weight for spectral loss
                 name="cvae", 
                 **kwargs):
        super(CVAE, self).__init__(name=name, **kwargs)
        
        # VAE components
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=input_dim)
        
        # Loss weights
        self.beta = beta  # Weight for KL term
        self.kl_weight = kl_weight  # Additional scaling for KL
        self.recon_weight = recon_weight  # Weight for reconstruction loss
        self.perceptual_weight = perceptual_weight  # Weight for perceptual loss
        self.spectral_weight = spectral_weight  # Weight for spectral loss
        
        # Tracking metrics during training
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.spectral_loss_tracker = keras.metrics.Mean(name="spectral_loss")
        
        # For debugging
        self.debug_counter = 0
    
    @property
    def metrics(self):
        metrics = [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.spectral_loss_tracker,
        ]
        return metrics
    
    def call(self, inputs, training=False):
        ppg, hr, br = inputs
        
        # Encode
        z_mean, z_log_var, z = self.encoder([ppg, hr, br], training=training)
        
        # Decode
        reconstructed = self.decoder([z, hr, br], training=training)
        
        # Debug info
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
            
        if self.debug_counter % 50 == 0 and training:
            tf.print(f"\nModel forward pass (sample {self.debug_counter}):")
            tf.print(f"  PPG input shape: {ppg.shape}")
            tf.print(f"  HR input: {hr[:5]}" + ("..." if tf.shape(hr)[0] > 5 else ""))
            tf.print(f"  BR input: {br[:5]}" + ("..." if tf.shape(br)[0] > 5 else ""))
            tf.print(f"  Latent space (z_mean) stats - Mean: {tf.reduce_mean(z_mean):.4f}, Std: {tf.math.reduce_std(z_mean):.4f}")
            tf.print(f"  Reconstructed PPG shape: {reconstructed.shape}")
            
            # Print sample values
            if tf.shape(ppg)[0] > 0:
                tf.print(f"  Original PPG sample (first 5 values): {ppg[0, :5]}")
                tf.print(f"  Reconstructed PPG sample (first 5 values): {reconstructed[0, :5]}")
        
        self.debug_counter += 1
        
        return reconstructed, z_mean, z_log_var, z
    
    def _compute_spectral_loss(self, x, x_recon):
        """
        Compute spectral loss using FFT to compare frequency domain representations
        """
        # Compute FFT
        x_fft = tf.abs(tf.signal.rfft(x))
        x_recon_fft = tf.abs(tf.signal.rfft(x_recon))
        
        # Focus on lower frequencies (more important for physiological signals)
        # Use weighting that decreases with frequency
        fft_len = tf.shape(x_fft)[1]
        freq_weights = tf.linspace(1.0, 0.1, fft_len)
        freq_weights = tf.reshape(freq_weights, [1, -1])
        
        # Apply frequency weighting and compute weighted MSE
        weighted_diff = freq_weights * tf.square(x_fft - x_recon_fft)
        spectral_loss = tf.reduce_mean(tf.reduce_sum(weighted_diff, axis=1))
        
        return spectral_loss
    
    def train_step(self, data):
        if len(data) == 3:
            ppg, hr, br = data
        else:
            raise ValueError("Expected input data to contain 3 items: ppg, hr, br")
            
        with tf.GradientTape() as tape:
            # Forward pass
            reconstructed, z_mean, z_log_var, z = self([ppg, hr, br], training=True)
            
            # Reconstruction loss (MSE) - manually calculated to avoid Keras version issues
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(ppg - reconstructed), axis=1
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )
            
            # Spectral loss - compare frequency components
            spectral_loss = self._compute_spectral_loss(ppg, reconstructed)
            
            # Total loss with weighting
            total_loss = (
                self.recon_weight * reconstruction_loss + 
                self.beta * self.kl_weight * kl_loss +
                self.spectral_weight * spectral_loss
            )
            
            # Manual weight decay if using standard Adam
            if not HAS_TFA and hasattr(self, 'optimizer') and isinstance(self.optimizer, tf.keras.optimizers.Adam):
                weight_decay = 1e-5  # Same as in AdamW
                for weight in self.trainable_weights:
                    if 'kernel' in weight.name or 'weight' in weight.name:
                        total_loss += weight_decay * tf.reduce_sum(tf.square(weight))
        
        # Calculate gradients and apply updates
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # Gradient clipping to prevent exploding gradients
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.spectral_loss_tracker.update_state(spectral_loss)
        
        # Return metrics
        results = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "spectral_loss": self.spectral_loss_tracker.result(),
        }
        return results
    
    def test_step(self, data):
        # Similar to train_step but without gradient calculation
        ppg, hr, br = data
        
        # Forward pass
        reconstructed, z_mean, z_log_var, z = self([ppg, hr, br], training=False)
        
        # Reconstruction loss (MSE) - manually calculated
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(ppg - reconstructed), axis=1
            )
        )
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
        )
        
        # Spectral loss
        spectral_loss = self._compute_spectral_loss(ppg, reconstructed)
        
        # Total loss with weighting
        total_loss = (
            self.recon_weight * reconstruction_loss + 
            self.beta * self.kl_weight * kl_loss +
            self.spectral_weight * spectral_loss
        )
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.spectral_loss_tracker.update_state(spectral_loss)
        
        # Return metrics
        results = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "spectral_loss": self.spectral_loss_tracker.result(),
        }
        return results
    
    def generate(self, hr, br, num_samples=1):
        """
        Generate PPG signal conditioned on HR and BR
        
        Args:
            hr: Heart Rate tensor [batch_size, 1]
            br: Breathing Rate tensor [batch_size, 1]
            num_samples: Number of samples to generate
            
        Returns:
            Generated PPG signals [num_samples, output_dim]
        """
        # Create random latent vectors
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        
        # Repeat HR and BR for each sample if needed
        if num_samples > 1 and tf.shape(hr)[0] == 1:
            hr = tf.repeat(hr, num_samples, axis=0)
            br = tf.repeat(br, num_samples, axis=0)
        
        # Generate PPG signals
        generated_ppg = self.decoder([z, hr, br], training=False)
        
        # Print info
        tf.print(f"\nGenerate PPG with conditions:")
        tf.print(f"  HR input: {hr}")
        tf.print(f"  BR input: {br}")
        tf.print(f"  Generated {num_samples} PPG signals with shape: {generated_ppg.shape}")
        
        if num_samples == 1:
            mean_val = tf.reduce_mean(generated_ppg)
            std_val = tf.math.reduce_std(generated_ppg)
            tf.print(f"  PPG output stats - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            tf.print(f"  PPG output sample (first 5 values): {generated_ppg[0, :5]}")
        
        return generated_ppg
    
    def reconstruct(self, ppg, hr, br):
        """
        Reconstruct PPG signal from input
        
        Args:
            ppg: Input PPG signal [batch_size, input_dim]
            hr: Heart Rate [batch_size, 1]
            br: Breathing Rate [batch_size, 1]
            
        Returns:
            Reconstructed PPG signal [batch_size, output_dim]
        """
        z_mean, z_log_var, z = self.encoder([ppg, hr, br], training=False)
        reconstructed_ppg = self.decoder([z, hr, br], training=False)
        return reconstructed_ppg
    
    def save_model(self, filepath):
        """Save the full model to file"""
        self.save(filepath)
    
    def load_model(self, filepath):
        """Load the full model from file"""
        self.load_weights(filepath)

# Custom callback để thay đổi beta (annealing)
class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, beta_start=0.0, beta_end=1.0, epochs=1000):
        super(BetaScheduler, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epochs = epochs
        self.beta_schedule = np.linspace(beta_start, beta_end, epochs)
    
    def on_epoch_begin(self, epoch, logs=None):
        new_beta = self.beta_schedule[epoch]
        self.model.beta = new_beta
        print(f"\nEpoch {epoch+1}: Setting beta to {new_beta:.4f}")

# Custom callback để lưu mô hình ở một số epoch
class SaveModelAtEpochs(keras.callbacks.Callback):
    def __init__(self, filepath_prefix, save_freq=100):
        super(SaveModelAtEpochs, self).__init__()
        self.filepath_prefix = filepath_prefix
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0 or epoch == self.params['epochs'] - 1:
            filepath = f"{self.filepath_prefix}_epoch_{epoch+1}.h5"
            self.model.save(filepath)
            print(f"\nSaved model to {filepath}")

# Hàm huấn luyện mô hình
def train_cvae(model, train_dataset, val_dataset, epochs=1000, 
               beta_start=0.0, beta_end=1.0, 
               early_stopping_patience=20, 
               learning_rate=0.001,
               weight_decay=1e-5):
    """
    Huấn luyện mô hình cVAE với các kỹ thuật tiên tiến
    """
    # Optimizer with weight decay and learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs//10,
        decay_rate=0.9,
        staircase=True
    )
    
    # Use AdamW optimizer if available, otherwise use Adam with custom weight decay
    if HAS_TFA:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay
        )
    else:
        # Use standard Adam with custom weight decay implementation
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model
    model.compile(optimizer=optimizer)
    
    # Callbacks
    callbacks = [
        # Beta annealing
        BetaScheduler(beta_start=beta_start, beta_end=beta_end, epochs=epochs),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/cvae_{time.strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1,
            write_graph=True
        ),
        
        # Save model periodically
        SaveModelAtEpochs(filepath_prefix='cvae_model', save_freq=100)
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Hàm sinh tín hiệu PPG với các giá trị HR và BR khác nhau
def generate_ppg_variations(model, hr_values, br_values):
    """
    Sinh tín hiệu PPG với các giá trị HR và BR khác nhau
    """
    print("\nSinh tín hiệu PPG với các giá trị HR và BR khác nhau")
    model.debug_counter = 0  # Reset debug counter
    
    # Tạo lưới 2D của các cặp HR/BR
    results = []
    
    for hr_val in hr_values:
        row = []
        for br_val in br_values:
            # Tạo tensor HR và BR
            hr_tensor = tf.constant([[hr_val]], dtype=tf.float32)
            br_tensor = tf.constant([[br_val]], dtype=tf.float32)
            
            # Sinh tín hiệu PPG
            generated_ppg = model.generate(hr_tensor, br_tensor)
            
            row.append({
                'hr': hr_val,
                'br': br_val,
                'ppg': generated_ppg[0].numpy()
            })
        results.append(row)
    
    return results

# Hàm chi tiết để in và đánh giá kết quả
def print_and_evaluate_results(model, test_dataset, num_samples=5):
    """
    In và đánh giá chi tiết kết quả của mô hình, so sánh đầu vào và đầu ra
    """
    model.debug_counter = 0  # Reset debug counter
    
    print("\n" + "="*80)
    print("CHI TIẾT ĐÁNH GIÁ MÔ HÌNH")
    print("="*80)
    
    # Lấy dữ liệu từ test_dataset
    test_samples = []
    for ppg, hr, br in test_dataset.take(1):
        test_samples = (ppg.numpy()[:num_samples], 
                       hr.numpy()[:num_samples], 
                       br.numpy()[:num_samples])
    
    ppg_samples, hr_samples, br_samples = test_samples
    
    print(f"\nĐÁNH GIÁ {num_samples} MẪU TỪ TẬP TEST:")
    
    # 1. TÁI TẠO PPG
    print("\n1. TÁI TẠO PPG (Reconstruction):")
    ppg_tensor = tf.convert_to_tensor(ppg_samples, dtype=tf.float32)
    hr_tensor = tf.convert_to_tensor(hr_samples, dtype=tf.float32)
    br_tensor = tf.convert_to_tensor(br_samples, dtype=tf.float32)
    
    recon_ppg = model.reconstruct(ppg_tensor, hr_tensor, br_tensor).numpy()
    
    for i in range(num_samples):
        hr_val = hr_samples[i][0]
        br_val = br_samples[i][0]
        
        # In thông tin về mẫu gốc
        print(f"\nMẫu #{i+1}:")
        print(f"  HR: {hr_val:.2f} BPM, BR: {br_val:.2f} BrPM")
        
        # Tính MSE cho tái tạo
        original_ppg = ppg_samples[i]
        reconstructed_ppg = recon_ppg[i]
        mse = np.mean((original_ppg - reconstructed_ppg) ** 2)
        
        print(f"  PPG gốc shape: {original_ppg.shape}")
        print(f"  PPG tái tạo shape: {reconstructed_ppg.shape}")
        print(f"  MSE tái tạo: {mse:.6f}")
        
        # In ra một số giá trị của PPG gốc và tái tạo
        print(f"  PPG gốc (5 giá trị đầu): {original_ppg[:5]}")
        print(f"  PPG tái tạo (5 giá trị đầu): {reconstructed_ppg[:5]}")
        print(f"  PPG gốc (5 giá trị cuối): {original_ppg[-5:]}")
        print(f"  PPG tái tạo (5 giá trị cuối): {reconstructed_ppg[-5:]}")
    
    # 2. TẠO PPG MỚI VỚI HR VÀ BR ĐÃ CHO
    print("\n\n2. TẠO PPG MỚI (Generation):")
    
    for i in range(num_samples):
        hr_val = hr_samples[i][0]
        br_val = br_samples[i][0]
        
        # Tạo PPG từ HR và BR đã cho
        hr_tensor = tf.convert_to_tensor([[hr_val]], dtype=tf.float32)
        br_tensor = tf.convert_to_tensor([[br_val]], dtype=tf.float32)
        
        # Tạo nhiều mẫu để kiểm tra tính nhất quán
        num_gen_samples = 3
        generated_ppgs = model.generate(hr_tensor, br_tensor, num_samples=num_gen_samples).numpy()
        
        print(f"\nSinh PPG với HR={hr_val:.2f} BPM, BR={br_val:.2f} BrPM:")
        
        for j in range(num_gen_samples):
            gen_ppg = generated_ppgs[j]
            print(f"  Mẫu sinh #{j+1} shape: {gen_ppg.shape}")
            print(f"  PPG sinh (5 giá trị đầu): {gen_ppg[:5]}")
            print(f"  PPG sinh (5 giá trị cuối): {gen_ppg[-5:]}")
        
        # Tính độ khác biệt giữa các mẫu sinh
        if num_gen_samples > 1:
            mse_between_samples = []
            for j in range(num_gen_samples):
                for k in range(j+1, num_gen_samples):
                    mse = np.mean((generated_ppgs[j] - generated_ppgs[k]) ** 2)
                    mse_between_samples.append(mse)
            
            print(f"  Độ khác biệt trung bình giữa các mẫu sinh (MSE): {np.mean(mse_between_samples):.6f}")
    
    # 3. KIỂM TRA TÍNH NHẤT QUÁN VỚI NHIỀU GIÁ TRỊ HR VÀ BR KHÁC NHAU
    print("\n\n3. KIỂM TRA TÍNH NHẤT QUÁN VỚI CÁC GIÁ TRỊ HR VÀ BR KHÁC NHAU:")
    
    # Tạo các giá trị HR và BR để kiểm tra
    test_hr_values = [60, 90, 120]  # BPM
    test_br_values = [12, 18, 24]   # BrPM
    
    results = {}
    
    for hr in test_hr_values:
        for br in test_br_values:
            hr_tensor = tf.constant([[hr]], dtype=tf.float32)
            br_tensor = tf.constant([[br]], dtype=tf.float32)
            
            # Sinh PPG
            gen_ppg = model.generate(hr_tensor, br_tensor)[0].numpy()
            
            # Lưu kết quả
            key = f"HR{hr}_BR{br}"
            results[key] = {
                'hr': hr,
                'br': br,
                'ppg': gen_ppg,
                'mean': np.mean(gen_ppg),
                'std': np.std(gen_ppg),
                'min': np.min(gen_ppg),
                'max': np.max(gen_ppg)
            }
            
            print(f"\nPPG sinh với HR={hr} BPM, BR={br} BrPM:")
            print(f"  Shape: {gen_ppg.shape}")
            print(f"  Mean: {results[key]['mean']:.4f}")
            print(f"  Std: {results[key]['std']:.4f}")
            print(f"  Min: {results[key]['min']:.4f}")
            print(f"  Max: {results[key]['max']:.4f}")
    
    # Kiểm tra xu hướng - HR cao hơn thường dẫn đến các đặc điểm khác trong PPG
    print("\n\nXU HƯỚNG THEO HR (BR cố định = 18):")
    for hr in test_hr_values:
        key = f"HR{hr}_BR18"
        if key in results:
            print(f"  HR={hr} BPM: Mean={results[key]['mean']:.4f}, Std={results[key]['std']:.4f}")
    
    print("\nXU HƯỚNG THEO BR (HR cố định = 90):")
    for br in test_br_values:
        key = f"HR90_BR{br}"
        if key in results:
            print(f"  BR={br} BrPM: Mean={results[key]['mean']:.4f}, Std={results[key]['std']:.4f}")
    
    print("\n" + "="*80)
    print("KẾT THÚC ĐÁNH GIÁ")
    print("="*80)
    return results

# Hàm trực quan hóa kết quả
def visualize_results(model, test_dataset):
    """
    Trực quan hóa kết quả tái tạo và sinh PPG
    """
    model.debug_counter = 0  # Reset debug counter
    
    # Lấy một batch từ tập test
    for ppg_samples, hr_samples, br_samples in test_dataset.take(1):
        break
    
    # Tái tạo PPG
    recon_ppg = model.reconstruct(ppg_samples, hr_samples, br_samples).numpy()
    
    # Hiển thị kết quả tái tạo
    plt.figure(figsize=(15, 10))
    
    for i in range(min(3, len(ppg_samples))):
        hr_val = hr_samples[i].numpy()[0]
        br_val = br_samples[i].numpy()[0]
        
        plt.subplot(3, 2, i*2 + 1)
        plt.plot(ppg_samples[i].numpy())
        plt.title(f"PPG gốc - HR: {hr_val:.1f}, BR: {br_val:.1f}")
        
        plt.subplot(3, 2, i*2 + 2)
        plt.plot(recon_ppg[i])
        plt.title("PPG tái tạo")
    
    plt.tight_layout()
    plt.savefig('ppg_reconstruction.png')
    plt.close()
    
    # Sinh PPG với các giá trị HR và BR khác nhau
    hr_values = [60, 80, 100, 120]  # Giá trị thực
    br_values = [10, 15, 20, 25]    # Giá trị thực
    
    generated_results = generate_ppg_variations(model, hr_values, br_values)
    
    # Hiển thị kết quả sinh
    plt.figure(figsize=(15, 15))
    
    for i, hr_row in enumerate(generated_results):
        for j, result in enumerate(hr_row):
            plt.subplot(len(hr_values), len(br_values), i*len(br_values) + j + 1)
            plt.plot(result['ppg'])
            plt.title(f"HR: {result['hr']:.1f}, BR: {result['br']:.1f}")
            plt.xticks([])
    
    plt.tight_layout()
    plt.savefig('ppg_generated_variations.png')
    plt.close()

# Hàm kiểm tra đặc tính của tín hiệu PPG theo HR và BR
def verify_ppg_characteristics(model):
    """
    Kiểm tra tính chất của tín hiệu PPG theo các giá trị HR và BR khác nhau
    Tạo một bảng phân tích để xác định mối quan hệ
    """
    model.debug_counter = 0  # Reset debug counter
    
    print("\n" + "="*80)
    print("KIỂM TRA ĐẶC TÍNH CỦA TÍN HIỆU PPG THEO HR VÀ BR")
    print("="*80)
    
    # Tạo các giá trị HR và BR để kiểm tra
    hr_values = [60, 70, 80, 90, 100, 110, 120]  # BPM
    br_values = [10, 12, 15, 18, 20, 22, 25]     # BrPM
    
    # Tạo ma trận kết quả
    results = {}
    
    # Tạo PPG cho tất cả các cặp HR/BR
    for hr in hr_values:
        for br in br_values:
            hr_tensor = tf.constant([[hr]], dtype=tf.float32)
            br_tensor = tf.constant([[br]], dtype=tf.float32)
            
            # Sinh 3 mẫu PPG cho mỗi cặp HR/BR và lấy trung bình
            num_samples = 3
            all_ppgs = model.generate(hr_tensor, br_tensor, num_samples=num_samples).numpy()
            ppg_samples = [all_ppgs[i] for i in range(num_samples)]
            
            # Tính các đặc trưng của tín hiệu PPG
            ppg_features = {}
            
            for i, ppg in enumerate(ppg_samples):
                # Tính số đỉnh (ước lượng nhịp tim)
                peaks, _ = find_peaks(ppg, height=0.5, distance=10)
                num_peaks = len(peaks)
                
                # Ước lượng HR từ PPG (số đỉnh * 60 / thời gian (giây))
                # Giả sử PPG dài 10 giây (1250 mẫu ở 125Hz)
                estimated_hr = num_peaks * 6  # * 60 / 10
                
                # Tính biên độ trung bình của đỉnh
                if len(peaks) > 0:
                    peak_amplitudes = ppg[peaks]
                    avg_peak_amplitude = peak_amplitudes.mean()
                else:
                    avg_peak_amplitude = 0
                
                # Tính các giá trị thống kê cơ bản
                ppg_mean = ppg.mean()
                ppg_std = ppg.std()
                ppg_min = ppg.min()
                ppg_max = ppg.max()
                
                # Phân tích tần số (FFT)
                N = len(ppg)
                T = 1.0 / 125.0  # sampling interval (125 Hz)
                
                yf = fft(ppg)
                xf = fftfreq(N, T)[:N//2]
                yf_abs = 2.0/N * np.abs(yf[:N//2])
                
                # Tìm tần số chính
                dominant_freq_idx = np.argmax(yf_abs[1:]) + 1  # bỏ qua DC (0 Hz)
                dominant_freq = xf[dominant_freq_idx]
                dominant_freq_amplitude = yf_abs[dominant_freq_idx]
                
                # Chuyển đổi tần số chính thành HR ước lượng (tần số * 60)
                dominant_freq_hr = dominant_freq * 60
                
                # Lưu kết quả
                ppg_features[i] = {
                    'num_peaks': num_peaks,
                    'estimated_hr': estimated_hr,
                    'avg_peak_amplitude': avg_peak_amplitude,
                    'mean': ppg_mean,
                    'std': ppg_std,
                    'min': ppg_min,
                    'max': ppg_max,
                    'dominant_freq': dominant_freq,
                    'dominant_freq_hr': dominant_freq_hr,
                    'dominant_freq_amplitude': dominant_freq_amplitude
                }
            
            # Tính trung bình các đặc trưng
            avg_features = {
                'num_peaks': np.mean([f['num_peaks'] for f in ppg_features.values()]),
                'estimated_hr': np.mean([f['estimated_hr'] for f in ppg_features.values()]),
                'avg_peak_amplitude': np.mean([f['avg_peak_amplitude'] for f in ppg_features.values()]),
                'mean': np.mean([f['mean'] for f in ppg_features.values()]),
                'std': np.mean([f['std'] for f in ppg_features.values()]),
                'dominant_freq_hr': np.mean([f['dominant_freq_hr'] for f in ppg_features.values()]),
            }
            
            # Tính độ sai lệch giữa HR đầu vào và HR ước lượng
            hr_error = abs(hr - avg_features['estimated_hr'])
            hr_error_percent = (hr_error / hr) * 100 if hr > 0 else 0
            
            # Lưu kết quả
            key = f"HR{hr}_BR{br}"
            results[key] = {
                'hr_input': hr,
                'br_input': br,
                'ppg_features': avg_features,
                'hr_error': hr_error,
                'hr_error_percent': hr_error_percent
            }
            
            print(f"\nPPG với HR={hr} BPM, BR={br} BrPM:")
            print(f"  Số đỉnh trung bình: {avg_features['num_peaks']:.1f}")
            print(f"  HR ước lượng từ số đỉnh: {avg_features['estimated_hr']:.1f} BPM")
            print(f"  HR ước lượng từ FFT: {avg_features['dominant_freq_hr']:.1f} BPM")
            print(f"  Sai số HR: {hr_error:.1f} BPM ({hr_error_percent:.1f}%)")
            print(f"  Biên độ đỉnh trung bình: {avg_features['avg_peak_amplitude']:.4f}")
            print(f"  Giá trị trung bình: {avg_features['mean']:.4f}")
            print(f"  Độ lệch chuẩn: {avg_features['std']:.4f}")
    
    # Tính độ chính xác trung bình
    avg_hr_error = np.mean([r['hr_error'] for r in results.values()])
    avg_hr_error_percent = np.mean([r['hr_error_percent'] for r in results.values()])
    
    print("\n" + "="*50)
    print(f"Sai số HR trung bình: {avg_hr_error:.2f} BPM ({avg_hr_error_percent:.2f}%)")
    
    # In xu hướng theo HR (BR cố định)
    fixed_br = 15
    print(f"\nXU HƯỚNG THEO HR (BR = {fixed_br}):")
    print(f"{'HR (BPM)':<10} {'HR ước lượng':<15} {'Sai số':<10} {'Biên độ':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 65)
    
    for hr in hr_values:
        key = f"HR{hr}_BR{fixed_br}"
        if key in results:
            r = results[key]
            features = r['ppg_features']
            print(f"{hr:<10} {features['estimated_hr']:<15.1f} {r['hr_error']:<10.1f} "
                  f"{features['avg_peak_amplitude']:<10.4f} {features['mean']:<10.4f} {features['std']:<10.4f}")
    
    # In xu hướng theo BR (HR cố định)
    fixed_hr = 80
    print(f"\nXU HƯỚNG THEO BR (HR = {fixed_hr}):")
    print(f"{'BR (BrPM)':<10} {'HR ước lượng':<15} {'Sai số':<10} {'Biên độ':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 65)
    
    for br in br_values:
        key = f"HR{fixed_hr}_BR{br}"
        if key in results:
            r = results[key]
            features = r['ppg_features']
            print(f"{br:<10} {features['estimated_hr']:<15.1f} {r['hr_error']:<10.1f} "
                  f"{features['avg_peak_amplitude']:<10.4f} {features['mean']:<10.4f} {features['std']:<10.4f}")
    
    print("\n" + "="*80)
    
    return results

# Hàm đánh giá hiệu năng sinh tín hiệu PPG
def generate_and_evaluate_ppg_signal(model, hr, br, num_samples=5):
    """
    Sinh tín hiệu PPG với HR và BR cụ thể và đánh giá chi tiết
    """
    model.debug_counter = 0  # Reset debug counter
    
    print(f"\n{'='*30} SINH VÀ ĐÁNH GIÁ TÍN HIỆU PPG {'='*30}")
    print(f"HR: {hr} BPM, BR: {br} BrPM")
    
    # Tạo tensor HR và BR
    hr_tensor = tf.constant([[hr]], dtype=tf.float32)
    br_tensor = tf.constant([[br]], dtype=tf.float32)
    
    # Sinh nhiều mẫu PPG
    ppg_signals = model.generate(hr_tensor, br_tensor, num_samples=num_samples).numpy()
    
    # In thông tin về tín hiệu PPG
    for i, ppg in enumerate(ppg_signals):
        print(f"\nMẫu PPG #{i+1}:")
        print(f"  Shape: {ppg.shape}")
        print(f"  Mean: {np.mean(ppg):.4f}")
        print(f"  Std: {np.std(ppg):.4f}")
        print(f"  Min: {np.min(ppg):.4f}")
        print(f"  Max: {np.max(ppg):.4f}")
        
        # Phân tích tần số
        N = len(ppg)
        T = 1.0 / 125.0  # sampling interval (125 Hz)
        
        yf = fft(ppg)
        xf = fftfreq(N, T)[:N//2]
        yf_abs = 2.0/N * np.abs(yf[:N//2])
        
        # Tìm tần số chính
        top_freqs_idx = np.argsort(yf_abs[1:])[-5:] + 1  # 5 tần số mạnh nhất (bỏ qua DC)
        
        print("  Top 5 tần số:")
        for idx in reversed(top_freqs_idx):
            freq = xf[idx]
            amp = yf_abs[idx]
            equivalent_hr = freq * 60  # Tần số * 60 = nhịp/phút
            print(f"    {freq:.2f} Hz (≈ {equivalent_hr:.1f} BPM): Amplitude = {amp:.4f}")
        
        # Tìm và đánh dấu các đỉnh
        peaks, _ = find_peaks(ppg, height=0.5, distance=10)
        
        num_peaks = len(peaks)
        estimated_hr = num_peaks * 6  # Giả sử tín hiệu dài 10 giây: * 60 / 10
        
        print(f"  Số đỉnh: {num_peaks}")
        print(f"  HR ước lượng từ số đỉnh: {estimated_hr:.1f} BPM")
        print(f"  Sai số HR: {abs(hr - estimated_hr):.1f} BPM ({abs(hr - estimated_hr)/hr*100:.1f}%)")
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 10))
    
    # Plot tín hiệu PPG
    plt.subplot(2, 1, 1)
    for i, ppg in enumerate(ppg_signals):
        plt.plot(ppg, label=f'Mẫu {i+1}')
        
        # Tìm và đánh dấu các đỉnh
        peaks, _ = find_peaks(ppg, height=0.5, distance=10)
        plt.plot(peaks, ppg[peaks], "x")
    
    plt.title(f'Tín hiệu PPG - HR: {hr} BPM, BR: {br} BrPM')
    plt.xlabel('Mẫu')
    plt.ylabel('Biên độ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot phổ tần số
    plt.subplot(2, 1, 2)
    for i, ppg in enumerate(ppg_signals):
        N = len(ppg)
        T = 1.0 / 125.0
        yf = fft(ppg)
        xf = fftfreq(N, T)[:N//2]
        yf_abs = 2.0/N * np.abs(yf[:N//2])
        
        plt.plot(xf, yf_abs, label=f'Mẫu {i+1}')
        
        # Đánh dấu tần số nhịp tim (HR/60 Hz)
        hr_freq = hr / 60
        plt.axvline(x=hr_freq, color='r', linestyle='--', alpha=0.5)
        
        # Đánh dấu tần số nhịp thở (BR/60 Hz)
        br_freq = br / 60
        plt.axvline(x=br_freq, color='g', linestyle='--', alpha=0.5)
    
    plt.title('Phổ tần số')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Biên độ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)  # Giới hạn hiển thị đến 5 Hz
    
    plt.tight_layout()
    plt.savefig(f'ppg_analysis_HR{hr}_BR{br}.png')
    plt.close()
    
    print(f"\nĐã lưu biểu đồ phân tích tại: ppg_analysis_HR{hr}_BR{br}.png")
    print("="*80)
    
    return ppg_signals

# Khởi tạo và huấn luyện mô hình
def main():
    if not patient_data:
        print("Không có dữ liệu để huấn luyện mô hình")
        return
    
    # Tạo dataset
    bidmc_dataset = BIDMCDataset(patient_data)
    
    # Chia tập huấn luyện và kiểm thử
    total_size = len(bidmc_dataset.ppg_segments)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # Tạo indexes ngẫu nhiên cho việc chia tập
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Tạo tập train
    train_ppg = bidmc_dataset.ppg_segments[train_indices]
    train_hr = bidmc_dataset.hrs[train_indices]
    train_br = bidmc_dataset.brs[train_indices]
    
    # Tạo tập test
    test_ppg = bidmc_dataset.ppg_segments[test_indices]
    test_hr = bidmc_dataset.hrs[test_indices]
    test_br = bidmc_dataset.brs[test_indices]
    
    # Tạo TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_ppg,
        train_hr.reshape(-1, 1),
        train_br.reshape(-1, 1)
    ))
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        test_ppg,
        test_hr.reshape(-1, 1),
        test_br.reshape(-1, 1)
    ))
    
    # Cấu hình datasets cho hiệu năng cao
    batch_size = 16  # Reduced batch size to avoid memory issues
    train_dataset = train_dataset.shuffle(buffer_size=train_size).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    
    print(f"\nTrain size: {train_size}, Test size: {test_size}")
    
    # Create model with improved architecture
    model = CVAE(
        latent_dim=128, 
        input_dim=1250,
        beta=0.0,  # Starting beta (will be scheduled during training)
        kl_weight=0.5,  # Reduced KL weight for better reconstruction
        recon_weight=1.5,  # Increased reconstruction weight
        spectral_weight=0.2  # Add spectral loss
    )
    
    # Carefully test model components separately to debug any issues
    for ppg_batch, hr_batch, br_batch in train_dataset.take(1):
        print(f"Testing model with sample batch: ppg:{ppg_batch.shape}, hr:{hr_batch.shape}, br:{br_batch.shape}")
        
        # Use a very small batch to ensure we catch issues early
        small_batch_size = 2
        small_ppg = ppg_batch[:small_batch_size]
        small_hr = hr_batch[:small_batch_size]
        small_br = br_batch[:small_batch_size]
        
        try:
            # Test encoder specifically
            print("\nTesting encoder...")
            z_mean, z_log_var, z = model.encoder([small_ppg, small_hr, small_br], training=False)
            print(f"Encoder output shapes: z_mean:{z_mean.shape}, z_log_var:{z_log_var.shape}, z:{z.shape}")
            
            # Test decoder separately
            print("\nTesting decoder with random latent vector...")
            random_z = tf.random.normal(shape=(small_batch_size, model.latent_dim))
            decoder_output = model.decoder([random_z, small_hr, small_br], training=False)
            print(f"Decoder output shape: {decoder_output.shape}")
            
            # Test full model
            print("\nTesting full model forward pass...")
            reconstructed, z_mean, z_log_var, z = model([small_ppg, small_hr, small_br], training=False)
            print(f"Full model output shapes: reconstructed:{reconstructed.shape}")
            
            # Test loss computation
            print("\nTesting loss computation...")
            with tf.GradientTape() as tape:
                reconstructed, z_mean, z_log_var, z = model([small_ppg, small_hr, small_br], training=True)
                
                # Manually calculate reconstruction loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(small_ppg - reconstructed), axis=1)
                )
                
                # KL divergence loss
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                
                # Spectral loss
                spectral_loss = model._compute_spectral_loss(small_ppg, reconstructed)
                
                # Total loss
                total_loss = reconstruction_loss + kl_loss + spectral_loss
            
            print(f"Loss values: recon={reconstruction_loss:.4f}, kl={kl_loss:.4f}, spectral={spectral_loss:.4f}")
            
            # Verify gradient computation
            grads = tape.gradient(total_loss, model.trainable_weights)
            grad_status = all(g is not None for g in grads)
            print(f"Gradient computation successful: {grad_status}")
            
            print("\nModel architecture testing successful!")
        except Exception as e:
            print(f"Error during model test: {e}")
            import traceback
            traceback.print_exc()
            print("Please fix model architecture before continuing.")
            return
        break
    
    # In tổng số tham số của mô hình
    model.summary()
    
    # Tạo callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Cấu hình optimizer với weight decay
    if HAS_TFA:
        optimizer = tfa.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=1e-5
        )
    else:
        # Use standard Adam and implement weight decay in training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile mô hình
    model.compile(optimizer=optimizer)
    
    # Huấn luyện mô hình
    print("\nĐang huấn luyện mô hình...")
    
    # Beta annealing schedule
    beta_scheduler = BetaScheduler(beta_start=0.0, beta_end=1.0, epochs=1000)
    
    # Lưu checkpoint - Fix filepath format for save_weights_only=True
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='cvae_checkpoint_epoch_{epoch:03d}.weights.h5',  # Must end with .weights.h5
        save_weights_only=True,
        save_freq=100 * (train_size // batch_size),
        verbose=1
    )
    
    # Huấn luyện mô hình
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=1000,
        callbacks=[
            beta_scheduler,
            early_stopping,
            checkpoint_callback,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs/cvae',
                histogram_freq=1
            )
        ],
        verbose=1
    )
    
    # Lưu mô hình cuối cùng
    model.save('cvae_model_final.h5')
    
    # In và đánh giá chi tiết kết quả
    print("\n*** ĐÁNH GIÁ CHI TIẾT ĐẦU VÀO VÀ ĐẦU RA ***")
    evaluation_results = print_and_evaluate_results(model, test_dataset, num_samples=5)
    
    # Trực quan hóa kết quả
    visualize_results(model, test_dataset)
    
    # Kiểm tra đặc tính của tín hiệu PPG theo HR và BR
    ppg_char_results = verify_ppg_characteristics(model)
    
    # Sinh và đánh giá chi tiết một số tín hiệu PPG
    # Chọn một số mẫu với HR và BR khác nhau
    test_cases = [(60, 12), (90, 18), (120, 24)]
    
    for hr, br in test_cases:
        ppg_signals = generate_and_evaluate_ppg_signal(model, hr, br)
    
    # Vẽ đồ thị mất mát
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_losses.png')
    plt.close()
    
    print("\nĐã huấn luyện mô hình cVAE và sinh tín hiệu PPG thành công!")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
