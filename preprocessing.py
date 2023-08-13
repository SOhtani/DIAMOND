import os
import csv
import chardet
import pandas as pd
from pathlib import Path

# 手順 1:指定した入力ディレクトリからcsvファイルをデコードして読み込む

input_dir = '/Users/jykim/Desktop/DIAMOND/DIAMOND/Data'
output_dir_base = '/Users/jykim/Desktop/DIAMOND/DIAMOND/output'

def load_and_process_csv_files(input_dir):
    airleak_data = []
    no_complication_data = []
    airleak_count = 0
    no_complication_count = 0
    preprocessed_file_list = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                if "no-complication" in root:  
                    label = 0
                elif "airleak" in root:
                    label = 1
                else:
                    continue
                
                try:
                    # エンコーディングを検出
                    with open(file_path, 'rb') as f:
                        result = chardet.detect(f.read())
                    encoding = result['encoding']
                    
                    # 検出されたエンコーディングが不正な場合、デフォルトのエンコーディング（'utf-8'）を使用
                    if encoding not in ['utf-8', 'utf-8-sig', 'ascii', 'MacRoman']:
                        encoding = 'utf-8'

                    # 検出されたエンコーディングでCSVファイルを読み込み
                    df = pd.read_csv(file_path, sep=';', encoding=encoding, parse_dates=['Date']).fillna(0)

                    elapsed_time = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds() / 3600
                    df['Elapsed Time'] = elapsed_time
                    drainage_period = elapsed_time.iloc[-1]

                    if label == 1:
                        output_subdir = "airleak"
                    elif label == 0:
                        output_subdir = "no-complication"
                    if drainage_period < 24:
                        output_subdir += "/too short drainage"
                    output_dir = os.path.join(output_dir_base, output_subdir)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
                    df.to_csv(output_file_path, index=False, sep=';')

                    filename = os.path.splitext(os.path.basename(output_file_path))[0]
                    preprocessed_file_list.append((filename, 'airleak' if label else 'no-complication'))

                    if label == 1:
                        airleak_data.append((df, label))
                        airleak_count += 1
                    elif label == 0:
                        no_complication_data.append((df, label))
                        no_complication_count += 1
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

    print(f"Loaded and processed {airleak_count + no_complication_count} CSV files.")
    return airleak_data, no_complication_data, airleak_count, no_complication_count, preprocessed_file_list

def save_preprocessed_file_list(preprocessed_file_list):
    df = pd.DataFrame(preprocessed_file_list, columns=['filename', 'label'])
    df.to_csv(os.path.join(output_dir_base, 'preprocessed_file_list.csv'), index=False)

# 手順 2f
def load_preprocessed_data():
    airleak_data = []
    no_complication_data = []
    short_drainage_count_airleak = 0
    short_drainage_count_no_complication = 0
    too_short_drainage_file_list = []
    for root, _, files in os.walk(output_dir_base):
        if "too short drainage" in root:
            if "no-complication" in root:
                short_drainage_count_no_complication += len(files)
            elif "airleak" in root:
                short_drainage_count_airleak += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                label = 'airleak' if "airleak" in root else 'no-complication'
                filename = os.path.splitext(os.path.basename(file_path))[0]  # extract filename without extension
                too_short_drainage_file_list.append((filename, label))
            continue

        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                if "no-complication" in root:
                    label = 0
                elif "airleak" in root:
                    label = 1
                else:
                    continue

                # データのロード
                df = pd.read_csv(file_path, sep=';', parse_dates=['Date'])
                 
                # 'Preset Pressure'が0または空白の行を除外
                df = df[df['Preset Pressure'] != 0]
                df = df.dropna(subset=['Preset Pressure'])
                
                elapsed_time = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds() / 3600
                df['Elapsed Time'] = elapsed_time
                
                # ファイル名の生成
                file_name_6h = f"til6h-{os.path.basename(file_path)}"
                file_name_12h = f"til12h-{os.path.basename(file_path)}"
                file_name_18h = f"til18h-{os.path.basename(file_path)}"
                file_name_24h = f"til24h-{os.path.basename(file_path)}"

                # 出力ディレクトリの設定
                output_subdir = "airleak" if label == 1 else "no-complication"

                # ファイルの保存
                for cutoff_hour, file_name in zip([6, 12, 18, 24], [file_name_6h, file_name_12h, file_name_18h, file_name_24h]):
                    df_capped = df[df['Elapsed Time'] <= cutoff_hour]
                    data_tuple = (df_capped, file_name)
                    
                    output_dir = os.path.join(output_dir_base, output_subdir, f"data until {cutoff_hour}h")
                    os.makedirs(output_dir, exist_ok=True)

                    output_file_path = os.path.join(output_dir, file_name)
                    df_capped.to_csv(output_file_path, index=False, sep=';')

                if label == 1:
                    airleak_data.append(df)
                elif label == 0:
                    no_complication_data.append(df)

    save_too_short_drainage_file_list(too_short_drainage_file_list)  # save the too short drainage file list
    return (airleak_data, no_complication_data, short_drainage_count_airleak, short_drainage_count_no_complication)

def save_too_short_drainage_file_list(too_short_drainage_file_list):
    df = pd.DataFrame(too_short_drainage_file_list, columns=['filename', 'label'])
    df.to_csv(os.path.join(output_dir_base, 'too_short_drainage_file_list.csv'), index=False)

def load_preprocessed_data_add():
    data_counts = {"6h": [0, 0], "12h": [0, 0], "18h": [0, 0], "24h": [0, 0]}
    
    for root, _, files in os.walk(output_dir_base):
        if "data until" in root:
            for cutoff_hour in ["6h", "12h", "18h", "24h"]:
                if f"data until {cutoff_hour}" in root:
                    if "no-complication" in root:
                        data_counts[cutoff_hour][1] += len(files)
                    elif "airleak" in root:
                        data_counts[cutoff_hour][0] += len(files)
    return data_counts

def main():
    airleak_data, no_complication_data, airleak_count, no_complication_count, preprocessed_file_list = load_and_process_csv_files(input_dir)
    save_preprocessed_file_list(preprocessed_file_list)
    
    (airleak_data, no_complication_data, short_drainage_count_airleak, short_drainage_count_no_complication) = load_preprocessed_data()

    print(f"Loaded files: airleak: {airleak_count}, no-complication: {no_complication_count}")
    print(f"Preprocessed files: airleak: {len(airleak_data)}, no-complication: {len(no_complication_data)}")
    print(f"Files with too short drainage period: airleak: {short_drainage_count_airleak}, no-complication: {short_drainage_count_no_complication}")
    
    data_counts = load_preprocessed_data_add()

    for cutoff_hour in ["6h", "12h", "18h", "24h"]:
        print(f"Data until {cutoff_hour}: airleak: {data_counts[cutoff_hour][0]}, no-complication: {data_counts[cutoff_hour][1]}")

# main関数を呼び出します
if __name__ == "__main__":
    main()