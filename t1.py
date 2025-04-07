import os
import pandas as pd

# Đường dẫn thư mục dữ liệu gốc và thư mục lưu kết quả
input_folder = r"data/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv"
output_folder = r"processed_data"

# Tạo thư mục processed_data nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)
processed_person_count = 0
# Số lượng người: 01 đến 53
for i in range(1, 54):
    full_processed = True

    # Định dạng số với 2 chữ số
    subj_id = f"{i:02d}"
    # Tạo thư mục con cho mỗi người
    subj_folder = os.path.join(output_folder, f"bidmc_{subj_id}")
    os.makedirs(subj_folder, exist_ok=True)
    
    # Đường dẫn file
    signals_file = os.path.join(input_folder, f"bidmc_{subj_id}_Signals.csv")
    numerics_file = os.path.join(input_folder, f"bidmc_{subj_id}_Numerics.csv")
    breaths_file = os.path.join(input_folder, f"bidmc_{subj_id}_Breaths.csv")
    
    # 1. Đọc và trích xuất từ Signals: chỉ lấy cột "Time [s]" và "PLETH"
    if os.path.exists(signals_file):
        df_signals = pd.read_csv(signals_file)
        df_signals.columns = df_signals.columns.str.strip()
        # Kiểm tra và lấy đúng tên cột (nếu có khoảng trắng, ...)
        if 'Time [s]' in df_signals.columns and 'PLETH' in df_signals.columns:
            df_signals = df_signals[['Time [s]', 'PLETH']]
        else:
            # Nếu không tìm thấy cột nào, đánh dấu là không đầy đủ
            full_processed = False
        # Lưu file đã xử lý
        df_signals.to_csv(os.path.join(subj_folder, "Signals_processed.csv"), index=False)
    
    # 2. Đọc và trích xuất từ Numerics: lấy cột "Time [s]", "HR" và "RESP"
    if os.path.exists(numerics_file):
        df_numerics = pd.read_csv(numerics_file)
        df_numerics.columns = df_numerics.columns.str.strip()

        # Kiểm tra và lấy đúng tên cột (nếu có khoảng trắng, ...)
        if 'Time [s]' in df_numerics.columns and 'HR' in df_numerics.columns and 'RESP' in df_numerics.columns:
            df_numerics = df_numerics[['Time [s]', 'HR', 'RESP']]
        else:
            # Nếu không tìm thấy cột nào, đánh dấu là không đầy đủ
            full_processed = False

        # Lưu file đã xử lý
        df_numerics.to_csv(os.path.join(subj_folder, "Numerics_processed.csv"), index=False)
    
    # 3. Đọc và trích xuất từ Breath: chọn cột annotator (ví dụ: đặt lại tên thành "ann1", "ann2")
    if os.path.exists(breaths_file):
        df_breaths = pd.read_csv(breaths_file)
        df_breaths.columns = df_breaths.columns.str.strip()
        # Kiểm tra và lấy đúng tên cột (nếu có khoảng trắng, ...)
        if 'breaths ann1 [signal sample no]' in df_breaths.columns and 'breaths ann2 [signal sample no]' in df_breaths.columns:
            # Đổi tên cột cho dễ đọc
            rename_dict = {}
            for col in df_breaths.columns:
                if "ann1" in col.lower():
                    rename_dict[col] = "ann1"
                elif "ann2" in col.lower():
                    rename_dict[col] = "ann2"
            if rename_dict:
                df_breaths.rename(columns=rename_dict, inplace=True)

            df_breaths = df_breaths[['ann1', 'ann2']]
        else:
            # Nếu không tìm thấy cột nào, đánh dấu là không đầy đủ
            full_processed = False
        # Giả sử tên cột ban đầu là "breaths ann1 [signal sample no]" và "breaths ann2 [signal sample no]"
        
        # Lưu file đã xử lý
        df_breaths.to_csv(os.path.join(subj_folder, "Breaths_processed.csv"), index=False)
    if not full_processed:
        print(f"Không đúng tên cột hoặc không đủ dữ liệu dữ liệu cho người {subj_id}. Bỏ qua.")
        continue
    processed_person_count += 1

# Thông báo đã xử lý xong một người
print(f"Đã xử lý dữ liệu cho {processed_person_count} người.")

# Sau khi xử lý xong, ta sẽ làm sạch dữ liệu trong các file đã lưu trong thư mục processed_data
print("Bắt đầu làm sạch dữ liệu...")

# Sau khi lưu dữ liệu, ta load lại dữ liệu từ processed_data để xử lý các giá trị null hoặc không hợp lệ
# Ví dụ: điền giá trị bị thiếu (fillna) với phương pháp forward fill, và sau đó loại bỏ nếu vẫn còn null.
def clean_dataframe(df, method='ffill'):
    # Áp dụng forward fill
    # df_clean = df.fillna(method=method)
    if method == 'ffill':
        df_clean = df.ffill()
    elif method == 'bfill':
        df_clean = df.bfill()
    else:
        raise ValueError("Chỉ hỗ trợ 'ffill' hoặc 'bfill'")
    # Nếu vẫn còn null, loại bỏ các dòng chứa null
    df_clean = df_clean.dropna()
    return df_clean

# Duyệt qua từng thư mục con và làm sạch dữ liệu
for subj in os.listdir(output_folder):
    subj_path = os.path.join(output_folder, subj)
    if os.path.isdir(subj_path):
        # Danh sách file đã xử lý trong thư mục của người đó
        for file_name in os.listdir(subj_path):
            if file_name.endswith("_processed.csv"):
                file_path = os.path.join(subj_path, file_name)
                df = pd.read_csv(file_path)
                df_clean = clean_dataframe(df)
                # Lưu lại file sau khi làm sạch (có thể ghi đè file cũ)
                df_clean.to_csv(file_path, index=False)
                
print("Đã trích xuất và làm sạch dữ liệu vào thư mục 'processed_data'.")

import matplotlib.pyplot as plt
import random

# Đường dẫn đến dữ liệu đã xử lý
base_dir = "processed_data"

# Tạo danh sách các subject IDs từ 01 đến 53
subject_ids = [f"{i:02d}" for i in range(1, 54)]

# Chọn ngẫu nhiên 3 subject
selected_subjects = random.sample(subject_ids, 3)

# Lặp qua từng subject được chọn
for subject in selected_subjects:
    # Tạo đường dẫn đến thư mục của subject
    subject_dir = os.path.join(base_dir, f"bidmc_{subject}")
    
    # Đọc các file CSV
    df_signals = pd.read_csv(os.path.join(subject_dir, 'Signals_processed.csv'))
    df_breaths = pd.read_csv(os.path.join(subject_dir, 'Breaths_processed.csv'))
    df_numerics = pd.read_csv(os.path.join(subject_dir, 'Numerics_processed.csv'))
    
    # Lấy đoạn tín hiệu PLETH từ 0 đến 30 giây
    pleth_segment = df_signals[df_signals['Time [s]'] <= 30]
    
    # Đảm bảo các sample number của breath annotations nằm trong phạm vi tín hiệu
    max_sample = len(df_signals) - 1
    valid_ann1 = df_breaths['ann1'][df_breaths['ann1'] <= max_sample]
    valid_ann2 = df_breaths['ann2'][df_breaths['ann2'] <= max_sample]
    
    # Chuyển sample number thành thời gian và lọc trong khoảng 0-30 giây
    breath_times_ann1 = df_signals['Time [s]'].iloc[valid_ann1]
    breath_times_ann1_segment = breath_times_ann1[breath_times_ann1 <= 30]
    
    breath_times_ann2 = df_signals['Time [s]'].iloc[valid_ann2]
    breath_times_ann2_segment = breath_times_ann2[breath_times_ann2 <= 30]
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Biểu đồ 1: PLETH signal với breath annotations
    ax1.plot(pleth_segment['Time [s]'], pleth_segment['PLETH'], label='PLETH', color='black')
    ax1.vlines(breath_times_ann1_segment, ymin=pleth_segment['PLETH'].min(), 
               ymax=pleth_segment['PLETH'].max(), colors='blue', linestyles='solid', label='Ann1')
    ax1.vlines(breath_times_ann2_segment, ymin=pleth_segment['PLETH'].min(), 
               ymax=pleth_segment['PLETH'].max(), colors='red', linestyles='dashed', label='Ann2')
    ax1.set_xlabel('Thời gian (giây)')
    ax1.set_ylabel('PLETH')
    ax1.legend()
    ax1.set_title(f'Subject bidmc_{subject} - Tín hiệu PLETH với nhịp thở (0-30 giây)')
    
    # Biểu đồ 2: HR và RESP
    ax2.plot(df_numerics['Time [s]'], df_numerics['HR'], color='green', label='HR')
    ax2.set_xlabel('Thời gian (giây)')
    ax2.set_ylabel('HR (nhịp/phút)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Tạo trục y thứ hai cho RESP
    ax2_r = ax2.twinx()
    ax2_r.plot(df_numerics['Time [s]'], df_numerics['RESP'], color='purple', label='RESP')
    ax2_r.set_ylabel('RESP (nhịp/phút)', color='purple')
    ax2_r.tick_params(axis='y', labelcolor='purple')
    
    # Thêm legend cho cả HR và RESP
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax2.set_title(f'Subject bidmc_{subject} - Nhịp tim (HR) và Nhịp thở (RESP)')
    
    # Điều chỉnh layout để tránh overlap
    plt.tight_layout()

# Hiển thị tất cả các biểu đồ
plt.show()

from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=8.0, fs=125.0, order=2):
    nyq = 0.5 * fs  # Tần số Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')  # Tạo bộ lọc bandpass
    return filtfilt(b, a, signal)  # Áp dụng lọc

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import butter, filtfilt

# Hàm bandpass filter cho tín hiệu PPG
def bandpass_filter(signal, lowcut=0.5, highcut=8.0, fs=125.0, order=2):
    nyq = 0.5 * fs  # Tần số Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Đường dẫn chung đến thư mục processed_data
base_dir = 'processed_data'

# Danh sách các thư mục bidmc_xx từ 01 đến 53
subject_ids = [f"{i:02d}" for i in range(1, 54)]  # Tạo danh sách từ '01' đến '53'

# Khởi tạo danh sách để lưu trữ segments và conditions từ tất cả các subjects
all_ppg_segments = []
all_hr_segments = []
all_resp_segments = []

# Số giây cho mỗi đoạn
segment_seconds = 10
segment_length = 125 * segment_seconds  # 1250 mẫu cho mỗi đoạn PPG

# Lặp qua từng subject
for subject in subject_ids:
    subject_dir = os.path.join(base_dir, f"bidmc_{subject}")
    
    # Đường dẫn đến file dữ liệu
    ppg_file = os.path.join(subject_dir, 'Signals_processed.csv')
    numeric_file = os.path.join(subject_dir, 'Numerics_processed.csv')
    
    # Kiểm tra file tồn tại
    if not os.path.exists(ppg_file) or not os.path.exists(numeric_file):
        print(f"File not found for subject {subject}, skipping...")
        continue
    
    # Đọc dữ liệu
    ppg_data = pd.read_csv(ppg_file)      # Giả sử có cột: 'Time [s]', 'PLETH'
    numeric_data = pd.read_csv(numeric_file)  # Giả sử có cột: 'Time [s]', 'HR', 'RESP'
    
    # Lọc tín hiệu PPG
    ppg_data['PLETH'] = bandpass_filter(ppg_data['PLETH'].values, lowcut=0.5, highcut=8.0, fs=125.0)
    
    # Số đoạn có thể cắt
    num_segments = len(ppg_data) // segment_length
    
    for t in range(num_segments):
        start_idx = t * segment_length
        end_idx = start_idx + segment_length
        if end_idx <= len(ppg_data):
            # Lấy đoạn PPG (1250 mẫu)
            ppg_segment = ppg_data['PLETH'].iloc[start_idx:end_idx].values
            
            # Với numerics, giả sử mẫu 1Hz nên mỗi đoạn có 10 mẫu
            start_time = t * segment_seconds
            end_time = start_time + segment_seconds
            hr_segment = numeric_data['HR'].iloc[start_time:end_time].values
            resp_segment = numeric_data['RESP'].iloc[start_time:end_time].values
            
            all_ppg_segments.append(ppg_segment)
            all_hr_segments.append(hr_segment)
            all_resp_segments.append(resp_segment)

# --- Normalization ---

# 1. Normalize toàn cục PPG
# Chuyển danh sách các đoạn PPG thành mảng numpy, shape = (total_segments, 1250)
ppg_all = np.array(all_ppg_segments)

# Fit một StandardScaler trên toàn bộ PPG (giữ mối tương quan giữa các đoạn)
ppg_scaler = StandardScaler()
# Lấy tất cả các mẫu từ tất cả các đoạn (reshape thành 2D)
ppg_scaler.fit(ppg_all.reshape(-1, 1))
# Transform từng đoạn và reshape lại về kích thước ban đầu
ppg_segments_normalized = ppg_scaler.transform(ppg_all.reshape(-1, 1)).reshape(ppg_all.shape)

print("PPG segments normalized shape:", ppg_segments_normalized.shape)

# 2. Normalize HR và RESP
# Tính trung bình của HR và RESP cho mỗi đoạn
hr_means = np.array([np.mean(seg) for seg in all_hr_segments])
resp_means = np.array([np.mean(seg) for seg in all_resp_segments])

# Dùng MinMaxScaler trên toàn bộ các giá trị trung bình
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
# Gộp lại để fit scaler, sau đó tách thành HR và RESP normalized
conditions_raw = np.stack([hr_means, resp_means], axis=1)
conditions_segments_normalized = minmax_scaler.fit_transform(conditions_raw)

print("Conditions segments normalized shape:", conditions_segments_normalized.shape)

# Bây giờ đầu ra của bạn:
# - ppg_segments_normalized: numpy array với shape (total_segments, 1250)
# - conditions_segments_normalized: numpy array với shape (total_segments, 2)

from sklearn.model_selection import train_test_split

X_train, X_test, C_train, C_test = train_test_split(ppg_segments_normalized, conditions_segments_normalized, test_size=0.2)

print(X_train.shape, C_train.shape)  # Kiểm tra kích thước dữ liệu huấn luyện
print(X_test.shape, C_test.shape)  # Kiểm tra kích thước dữ liệu kiểm tra

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=1250, condition_dim=2, latent_dim=32):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        
        # Giản lược CNN chỉ còn 2 lớp
        self.conv1 = nn.Conv1d(1, 16, 7, stride=4, padding=3)  # (1250 -> 313)
        self.conv2 = nn.Conv1d(16, 32, 5, stride=2, padding=2)  # (313 -> 157)
        
        # Tính toán kích thước sau CNN
        self.flatten_size = 32 * 157
        
        # Thêm nhiều lớp Dense hơn
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Latent space
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, condition):
        # Xử lý CNN
        x = x.view(-1, 1, self.input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Kết hợp condition và qua FC
        x = x.view(x.size(0), -1)
        x = torch.cat([x, condition], dim=1)
        x = self.fc_layers(x)
        
        return self.fc_mean(x), self.fc_logvar(x)
    

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, condition_dim=2, output_dim=1250):
        super(Decoder, self).__init__()
        # Mở rộng với nhiều lớp Dense
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 157)  # Khớp với Encoder output
        )
        
        # Điều chỉnh tham số deconv
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2),  # 157 -> 313
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 7, stride=4, padding=3, output_padding=1)  # 313 -> 1250
        )

    def forward(self, z, condition):
        # Xử lý FC
        x = torch.cat([z, condition], dim=1)
        x = self.fc_layers(x).view(-1, 32, 157)  # Khớp shape với Encoder
        
        # Xử lý deconv
        x = self.deconv_layers(x)
        return x.view(-1, 1250)  # Đảm bảo output_dim chính xác
    

class CVAE(nn.Module):
    def __init__(self, input_dim=1250, condition_dim=2, latent_dim=32):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, condition_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, input_dim)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, condition):
        mean, logvar = self.encoder(x, condition)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z, condition)
        return x_recon, mean, logvar
    

# Test Decoder
decoder = Decoder()
z = torch.randn(64, 32)
condition = torch.randn(64, 2)
output = decoder(z, condition)
print(output.shape)  # Phải ra torch.Size([64, 1250])

# Test full model
model = CVAE()
x = torch.randn(64, 1250)
cond = torch.randn(64, 2)
recon, _, _ = model(x, cond)
print(recon.shape)  # Phải ra torch.Size([64, 1250])


def loss_function(x_recon, x, mean, logvar, beta=2):
    mse = F.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return mse + beta * kl, mse, kl, beta


from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
import numpy as np

# Chuyển dữ liệu thành tensor
X_train_tensor = torch.FloatTensor(X_train)
C_train_tensor = torch.FloatTensor(C_train)
X_test_tensor = torch.FloatTensor(X_test)
C_test_tensor = torch.FloatTensor(C_test)

# Tạo DataLoader
train_dataset = TensorDataset(X_train_tensor, C_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, C_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Khởi tạo mô hình và optimizer
latent_dim = 32
cvae = CVAE(input_dim=segment_length, condition_dim=2, latent_dim=latent_dim)
optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)

# Lưu trữ loss
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_path = 'best_cvae_model.pth'

train_kl_losses = []
test_kl_losses = []
beta_history = []

num_epochs = 500
warmup_epochs = 10      # Epoch warm-up (beta = 0)
anneal_duration = 40    # Số epoch tăng beta từ 0 đến 1 sau warm-up
patience = 50
no_improve = 0

# Bắt đầu training
for epoch in range(num_epochs):
    # Tính beta với KL annealing
    if epoch < warmup_epochs:
        beta_val = 0.0
    else:
        beta_val = min(1.0, (epoch - warmup_epochs + 1) / anneal_duration)
    beta_history.append(beta_val)
    
    cvae.train()
    total_train_loss = 0.0
    total_train_kl_loss = 0.0
    total_train_mse_loss = 0.0
    num_batches = 0
    
    # Dùng tqdm để hiển thị progress bar
    for x, condition in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False):
        optimizer.zero_grad()
        x_recon, mean, logvar = cvae(x, condition)
        # loss_function trả về (loss, mse, kl, beta_out)
        loss, mse, kl, _ = loss_function(x_recon, x, mean, logvar, beta=beta_val)
        loss.backward()
        # (Tùy chọn) Gradient clipping: torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        total_train_kl_loss += kl.item()
        total_train_mse_loss += mse.item()
        num_batches += 1
    
    avg_train_loss = total_train_loss / num_batches
    avg_train_kl_loss = total_train_kl_loss / num_batches
    avg_train_mse_loss = total_train_mse_loss / num_batches
    train_losses.append(avg_train_loss)
    train_kl_losses.append(avg_train_kl_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, MSE: {avg_train_mse_loss:.4f}, KL: {avg_train_kl_loss:.4f}, Beta: {beta_val:.4f}")
    
    # Đánh giá trên tập test
    cvae.eval()
    total_test_loss = 0.0
    total_test_kl_loss = 0.0
    total_test_mse_loss = 0.0
    num_test_batches = 0
    with torch.no_grad():
        for x, condition in test_loader:
            x_recon, mean, logvar = cvae(x, condition)
            loss, mse, kl, _ = loss_function(x_recon, x, mean, logvar, beta=beta_val)
            total_test_loss += loss.item()
            total_test_kl_loss += kl.item()
            total_test_mse_loss += mse.item()
            num_test_batches += 1
    
    avg_test_loss = total_test_loss / num_test_batches
    avg_test_kl_loss = total_test_kl_loss / num_test_batches
    avg_test_mse_loss = total_test_mse_loss / num_test_batches
    test_losses.append(avg_test_loss)
    test_kl_losses.append(avg_test_kl_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}, MSE: {avg_test_mse_loss:.4f}, KL: {avg_test_kl_loss:.4f}")
    
    # Lưu mô hình tốt nhất theo test loss
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(cvae.state_dict(), best_model_path)
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print("Early stopping triggered!")
        break

# Tải mô hình tốt nhất
cvae_loaded = CVAE(input_dim=segment_length, condition_dim=2, latent_dim=latent_dim)
cvae_loaded.load_state_dict(torch.load(best_model_path))
cvae_loaded.eval()

# Vẽ đồ thị loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss during Training')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_kl_losses, label='Train KL Loss')
plt.plot(test_kl_losses, label='Test KL Loss')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.title('KL Loss during Training')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(beta_history, label='Beta (KL Annealing)', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Beta')
plt.title('Beta Schedule (KL Annealing)')
plt.grid(True)
plt.legend()
plt.show()


import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Đặt mô hình ở chế độ eval
cvae_loaded.eval()

# Tạo danh sách để lưu latent vectors và nhãn (HR, RESP)
z_list = []
hr_list = []
resp_list = []

# Duyệt qua tập test và thu thập mean (z) từ encoder
with torch.no_grad():
    for x, condition in test_loader:
        _, mean, _ = cvae_loaded(x, condition)
        z_list.append(mean)
        # print(mean.shape)
        hr_list.append(condition[:, 0])     # HR
        resp_list.append(condition[:, 1])   # RESP

# Ghép lại thành tensor
z_all = torch.cat(z_list, dim=0).cpu().numpy()
print("z_all.shape:", z_all.shape)
hr_all = torch.cat(hr_list, dim=0).cpu().numpy()
print("hr_all.shape:", hr_all.shape)
resp_all = torch.cat(resp_list, dim=0).cpu().numpy()
print("resp_all.shape:", resp_all.shape)

# Dùng PCA để giảm từ latent_dim về 2D
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_all)

# Vẽ scatter plot, tô màu theo HR hoặc RESP
plt.figure(figsize=(10, 5))

# Màu theo HR
plt.subplot(1, 2, 1)
plt.scatter(z_pca[:, 0], z_pca[:, 1], c=hr_all, cmap='coolwarm', s=10)
plt.colorbar(label='HR')
plt.title('Latent space - colored by HR')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

# Màu theo RESP
plt.subplot(1, 2, 2)
plt.scatter(z_pca[:, 0], z_pca[:, 1], c=resp_all, cmap='viridis', s=10)
plt.colorbar(label='RESP')
plt.title('Latent space - colored by RESP')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)

plt.tight_layout()
plt.show()

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Giả sử cvae_loaded là mô hình đã huấn luyện và test_loader có dữ liệu x và condition
means = []
with torch.no_grad():
    for x, condition in test_loader:
        # Lấy đầu ra từ encoder: sử dụng output "mean" của q(z|x)
        _, mean, _ = cvae_loaded(x, condition)
        means.append(mean)
        
means = torch.cat(means, dim=0).cpu().numpy()

# Vẽ histogram cho thành phần thứ 1 của vector mean
plt.figure(figsize=(8,4))
plt.hist(means[:, 0], bins=50, density=True, alpha=0.6, color='b', label='Encoder Mean dim 0')

# Vẽ thêm histogram của N(0,1)
x_vals = np.linspace(-4, 4, 100)
pdf = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * x_vals**2)
plt.plot(x_vals, pdf, 'r-', label='N(0,1) PDF')
plt.title('So sánh Histogram thành phần đầu của latent vector')
plt.xlabel('Giá trị')
plt.ylabel('Density')
plt.legend()
plt.show()

# Hoặc dùng PCA để giảm chiều và scatter plot
pca = PCA(n_components=2)
means_2d = pca.fit_transform(means)

plt.figure(figsize=(8,6))
plt.scatter(means_2d[:, 0], means_2d[:, 1], alpha=0.6, label='Encoder Latent Means')
plt.title('Phân bố 2D của latent vector từ Encoder (sau PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.show()

# Đánh giá trên tập test (MSE trung bình)
cvae_loaded.eval()
total_mse = 0
with torch.no_grad():
    for x, condition in test_loader:
        x_recon, _, _ = cvae_loaded(x, condition)
        mse = F.mse_loss(x_recon, x, reduction='mean')
        total_mse += mse.item()
avg_mse = total_mse / len(test_loader)
print(f"Average MSE on Test Set: {avg_mse:.4f}")

# So sánh PPG gốc và tái tạo
plt.figure(figsize=(10, 4))
plt.plot(X_test[0], label='PPG gốc')
plt.plot(x_recon[0].numpy(), label='PPG tái tạo')
plt.legend()
plt.title("So sánh PPG gốc và tái tạo")
plt.show()


import matplotlib.pyplot as plt
import torch

# Số mẫu muốn hiển thị
num_samples = 5

# Lấy nhiều mẫu từ tập test
x_originals = X_test_tensor[:num_samples]
c_originals = C_test_tensor[:num_samples]

# Inverse transform để lấy lại HR và RESP gốc
conditions_denorm = minmax_scaler.inverse_transform(c_originals.numpy())

# Khởi tạo danh sách kết quả
x_recons = []
x_gens = []

# Tái tạo và tạo sinh với mỗi điều kiện
cvae_loaded.eval()
with torch.no_grad():
    for i in range(num_samples):
        x = x_originals[i].unsqueeze(0)  # Thêm chiều batch
        c = c_originals[i].unsqueeze(0)

        # Tái tạo
        x_recon, _, _ = cvae_loaded(x, c)
        x_recons.append(x_recon[0])

        # Tạo latent vector ngẫu nhiên và tạo sinh
        z = torch.randn(1, latent_dim)
        x_gen = cvae_loaded.decoder(z, c)
        x_gens.append(x_gen[0])

# Vẽ
plt.figure(figsize=(12, num_samples * 3))
for i in range(num_samples):
    hr, resp = c_originals[i]
    plt.subplot(num_samples, 1, i + 1)
    plt.plot(x_originals[i].numpy(), label='PPG gốc', color='blue')
    plt.plot(x_recons[i].numpy(), label='PPG tái tạo', color='red', linestyle='--')
    plt.plot(x_gens[i].numpy(), label='PPG tạo sinh', color='green', linestyle='-.')
    plt.title(f'Mẫu {i+1} - HR={hr:.2f}, RESP={resp:.2f}')
    plt.xlabel('Thời gian')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


import torch
import matplotlib.pyplot as plt

# Số mẫu PPG muốn tạo
num_generated = 5

# Tạo latent vector ngẫu nhiên
z = torch.randn(num_generated, latent_dim)

# Tạo điều kiện HR và RESP gốc
conditions_generated_raw = np.array([
    [60, 12],
    [70, 14],
    [80, 16],
    [90, 18],
    [100, 20]
])

# Chuẩn hóa giống với conditions
conditions_generated_norm = minmax_scaler.transform(conditions_generated_raw)
conditions_generated = torch.FloatTensor(conditions_generated_norm)

# Tạo PPG mới
cvae_loaded.eval()
with torch.no_grad():
    ppg_generated = cvae_loaded.decoder(z, conditions_generated)

# Vẽ các PPG tạo sinh
plt.figure(figsize=(15, 10))
for i in range(num_generated):
    hr, resp = conditions_generated_raw[i]
    plt.subplot(num_generated, 1, i+1)
    plt.plot(ppg_generated[i].numpy(), label=f'PPG tạo sinh: HR={hr}, RESP={resp}')
    plt.legend()
    plt.xlabel('Thời gian')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()


