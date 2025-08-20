import os
import numpy as np
import h5py
from tqdm import tqdm
import glob

def find_h5_files(root_dir):
    """查找目录下所有的h5文件"""
    h5_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def load_h5_data(h5_path):
    """加载h5格式的点云数据"""
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"Successfully opened {h5_path}")
            print(f"Available keys: {list(f.keys())}")
            points = f['data'][:]
            labels = f['label'][:]
            return points, labels.squeeze()
    except Exception as e:
        print(f"Error loading {h5_path}: {str(e)}")
        return None, None

def process_modelnet40(input_root, output_root, num_points=1024):
    """
    将ModelNet40数据集转换为Few-Shot Learning所需格式
    """
    os.makedirs(output_root, exist_ok=True)
    
    # 首先查找所有h5文件
    print("Searching for h5 files...")
    h5_files = find_h5_files(input_root)
    print(f"Found {len(h5_files)} h5 files:")
    for f in h5_files:
        print(f"  {f}")
    
    if not h5_files:
        raise FileNotFoundError("No h5 files found in the directory!")
    
    # 收集所有数据
    all_points = []
    all_labels = []
    
    # 处理所有h5文件
    print("\nLoading data from h5 files...")
    for h5_file in tqdm(h5_files):
        points, labels = load_h5_data(h5_file)
        if points is not None and labels is not None:
            all_points.append(points)
            all_labels.append(labels)
    
    if not all_points:
        raise Exception("No data was loaded. Please check the file paths and data structure.")
    
    # 合并所有数据
    print("\nMerging all data...")
    all_points = np.concatenate(all_points, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total points shape: {all_points.shape}")
    print(f"Total labels shape: {all_labels.shape}")
    print(f"Unique labels: {np.unique(all_labels)}")
    
    # 为每个类别创建目录并保存点云数据
    print("\nSaving processed point clouds...")
    for class_idx in tqdm(range(40)):
        # 创建类别目录
        class_dir = os.path.join(output_root, str(class_idx))
        os.makedirs(class_dir, exist_ok=True)
        
        # 获取当前类别的所有样本
        class_mask = (all_labels == class_idx)
        class_points = all_points[class_mask]
        
        print(f"\nProcessing class {class_idx}:")
        print(f"  Found {len(class_points)} samples")
        
        # 保存每个样本
        for i, points in enumerate(class_points):
            # 随机采样到指定点数
            if len(points) > num_points:
                indices = np.random.choice(len(points), num_points, replace=False)
                points = points[indices]
            
            # 保存为npy文件
            save_path = os.path.join(class_dir, f'sample_{i:04d}.npy')
            np.save(save_path, points)

def main():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 配置路径
    input_root = './modelnet40_ply_hdf5_2048'
    output_root = './modelnet40_fs_crossvalidation'
    
    # 检查输入目录
    print(f"Checking input directory: {input_root}")
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input directory {input_root} does not exist!")
    
    # 列出目录内容
    print("\nDirectory contents:")
    for root, dirs, files in os.walk(input_root):
        level = root.replace(input_root, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
    
    # 处理数据集
    print(f"\nConverting ModelNet40 dataset from {input_root} to {output_root}")
    process_modelnet40(input_root, output_root)
    
    # 输出数据集信息
    print("\nChecking processed dataset...")
    class_counts = []
    for i in range(40):
        class_dir = os.path.join(output_root, str(i))
        if os.path.exists(class_dir):
            count = len(os.listdir(class_dir))
            class_counts.append(count)
            print(f"Class {i}: {count} samples")
    
    if class_counts:
        print("\nDataset Summary:")
        print(f"Total classes: {len(class_counts)}")
        print(f"Total samples: {sum(class_counts)}")
        print(f"Average samples per class: {np.mean(class_counts):.1f}")
        print(f"Min samples per class: {min(class_counts)}")
        print(f"Max samples per class: {max(class_counts)}")
    else:
        print("No data was processed!")

if __name__ == '__main__':
    main()
