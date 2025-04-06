import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_dataset(file_path):
    """
    Tải dữ liệu từ file (hỗ trợ .csv, .mat)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.mat':
        return loadmat(file_path)
    else:
        raise ValueError(f"Định dạng file {file_ext} không được hỗ trợ")

def extract_signals(data, signal_names=['PPG', 'HR', 'BR']):
    """
    Trích xuất các tín hiệu theo tên từ dữ liệu
    """
    extracted_data = {}
    
    # Xác định và trích xuất tín hiệu từ dữ liệu
    # Phần này cần điều chỉnh dựa vào cấu trúc thực tế của bộ dữ liệu
    if isinstance(data, pd.DataFrame):
        for signal in signal_names:
            if signal in data.columns:
                extracted_data[signal] = data[signal].values
            else:
                # Tìm kiếm cột có chứa tên tín hiệu
                matching_cols = [col for col in data.columns if signal.lower() in col.lower()]
                if matching_cols:
                    extracted_data[signal] = data[matching_cols[0]].values
    elif isinstance(data, dict):  # Cho file .mat
        for signal in signal_names:
            if signal in data:
                extracted_data[signal] = data[signal]
            else:
                # Tìm kiếm key có chứa tên tín hiệu
                matching_keys = [key for key in data.keys() if isinstance(key, str) and signal.lower() in key.lower()]
                if matching_keys:
                    extracted_data[signal] = data[matching_keys[0]]
    
    return extracted_data

def merge_datasets(dataset1, dataset2, signals=['PPG', 'HR', 'BR']):
    """
    Gộp hai bộ dữ liệu và chỉ giữ lại các tín hiệu được chỉ định
    """
    # Trích xuất tín hiệu từ mỗi bộ dữ liệu
    signals1 = extract_signals(dataset1, signals)
    signals2 = extract_signals(dataset2, signals)
    
    merged_data = {}
    
    # Gộp tín hiệu từ hai bộ dữ liệu
    for signal in signals:
        if signal in signals1 and signal in signals2:
            # Điều chỉnh độ dài nếu cần thiết (ví dụ: resampling)
            # Trong ví dụ này, ta chỉ nối trực tiếp
            merged_data[signal] = np.concatenate([signals1[signal], signals2[signal]])
        elif signal in signals1:
            merged_data[signal] = signals1[signal]
        elif signal in signals2:
            merged_data[signal] = signals2[signal]
    
    return merged_data

def save_merged_data(merged_data, output_file):
    """
    Lưu dữ liệu đã gộp vào file (hỗ trợ .csv, .mat)
    """
    file_ext = os.path.splitext(output_file)[1].lower()
    
    if file_ext == '.csv':
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(merged_data)
        df.to_csv(output_file, index=False)
    elif file_ext == '.mat':
        # Lưu dưới dạng file .mat
        from scipy.io import savemat
        savemat(output_file, merged_data)
    else:
        raise ValueError(f"Định dạng file {file_ext} không được hỗ trợ để lưu")

def visualize_signals(data, signals=['PPG', 'HR', 'BR']):
    """
    Hiển thị các tín hiệu trên đồ thị
    """
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 4*len(signals)))
    
    for i, signal in enumerate(signals):
        if signal in data:
            if len(signals) > 1:
                ax = axes[i]
            else:
                ax = axes
                
            ax.plot(data[signal])
            ax.set_title(f'Tín hiệu {signal}')
            ax.set_xlabel('Thời gian (mẫu)')
            ax.set_ylabel('Biên độ')
    
    plt.tight_layout()
    plt.show()

# Ví dụ sử dụng:
if __name__ == "__main__":
    # Đường dẫn đến các file dữ liệu
    bmidc_file = "path_to_bmidc_dataset.csv"  # hoặc .mat
    capno_file = "path_to_capno_ieee_dataset.csv"  # hoặc .mat
    output_file = "merged_signals.csv"  # hoặc .mat
    
    # Tải dữ liệu
    print("Đang tải dữ liệu...")
    bmidc_data = load_dataset(bmidc_file)
    capno_data = load_dataset(capno_file)
    
    # Gộp và trích xuất tín hiệu
    print("Đang gộp và trích xuất tín hiệu...")
    merged_data = merge_datasets(bmidc_data, capno_data, signals=['PPG', 'HR', 'BR'])
    
    # Lưu dữ liệu đã gộp
    print("Đang lưu dữ liệu đã gộp...")
    save_merged_data(merged_data, output_file)
    
    # Hiển thị tín hiệu
    print("Hiển thị tín hiệu...")
    visualize_signals(merged_data)
    
    print(f"Hoàn tất! Dữ liệu đã được lưu vào {output_file}")
