import pandas as pd
import random
from scipy.io import loadmat

def generate_csvs(num_data, percentage):
    # 1. Generate a list of numbers from 0 to n-1
    numbers = list(range(num_data))
    
    # 2. Shuffle these numbers randomly
    random.shuffle(numbers)
    
    # Calculate the split point
    split_point = int(num_data * percentage)
    
    # 3. Split the list
    train_idx = numbers[:split_point]
    test_idx = numbers[split_point:]
    
    # 4. Convert lists to pandas dataframes
    df1 = pd.DataFrame(train_idx, columns=["data_idx"])
    df2 = pd.DataFrame(test_idx, columns=["data_idx"])
    
    # 5. Save the dataframes to CSV files
    return df1, df2

if __name__ == "__main__":
    random.seed(42)

    data_path = "DeepMIMO/DeepMIMO_datasets/O1_3p5/channel_ad_clip.mat"
    data = loadmat(data_path)['all_channel_ad_clip']
    num_data = data.shape[0]
    percentage = 0.8
    (df1, df2) = generate_csvs(num_data, percentage)

    df1.to_csv('/'.join(data_path.split('/')[:-1])+'/train_data_idx.csv', index=False)
    df2.to_csv('/'.join(data_path.split('/')[:-1])+'/test_data_idx.csv', index=False)
