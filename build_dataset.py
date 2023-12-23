
import os
import shutil
import random
from tqdm import tqdm

# 定义移动图片函数--专门为了处理DDSM数据集
def move_images(source_folder, target_folder):
    for label in os.listdir(source_folder):
        label_folder = os.path.join(source_folder, label)
        if os.path.isdir(label_folder):
            target_label_folder = os.path.join(target_folder, label)
            os.makedirs(target_label_folder, exist_ok=True)

            for subfolder in os.listdir(label_folder):
                subfolder_path = os.path.join(label_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    for case_folder in os.listdir(subfolder_path):
                        case_folder_path = os.path.join(subfolder_path, case_folder)
                        if os.path.isdir(case_folder_path):
                            for root, dirs, files in os.walk(case_folder_path):
                                for file in files:
                                    if file.endswith(('.jpg', '.jpeg', '.png')):
                                        source_path = os.path.join(root, file)
                                        target_path = os.path.join(target_label_folder, file)

                                        shutil.move(source_path, target_path)
                                        print(f'Moved {file} to {label} folder.')
    print('OK!')

# 定义创建数据集函数
def create_imagenet_dataset(data_dir, target_dir, train_split_ratio, val_split_ratio, test_split_ratio):

    # 创建目标数据集文件夹及其子目录结构
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'meta'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
    # 获取原始数据文件夹下的子目录列表
    categories = os.listdir(data_dir)
    # 遍历每个子目录
    for category in categories:
        # 获取该类别下的所有文件
        files = os.listdir(os.path.join(data_dir, category))

        # 随机打乱文件顺序
        random.shuffle(files)

        # 计算划分数据集的索引
        total_files = len(files)
        train_split = int(train_split_ratio * total_files)
        val_split = int(val_split_ratio * total_files)

        # 划分数据集并复制到目标文件夹，使用tqdm添加进度条
        for file in tqdm(files[:train_split], desc=f'Copying train data for {category}'):
            src = os.path.join(data_dir, category, file)
            dst = os.path.join(target_dir, 'train', category)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, file))

        for file in tqdm(files[train_split:train_split + val_split], desc=f'Copying validation data for {category}'):
            src = os.path.join(data_dir, category, file)
            dst = os.path.join(target_dir, 'val', category)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, file))

        for file in tqdm(files[train_split + val_split:], desc=f'Copying test data for {category}'):
            src = os.path.join(data_dir, category, file)
            dst = os.path.join(target_dir, 'test', category)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, file))

    # 创建标注文件（train.txt、val.txt、test.txt）
    with open(os.path.join(target_dir, 'meta', 'train.txt'), 'w') as train_txt:
        for category in categories:
            train_files = os.listdir(os.path.join(target_dir, 'train', category))
            for file in train_files:
                train_txt.write(f'{os.path.join("train", category, file)} {category}\n')

    with open(os.path.join(target_dir, 'meta', 'val.txt'), 'w') as val_txt:
        for category in categories:
            val_files = os.listdir(os.path.join(target_dir, 'val', category))
            for file in val_files:
                val_txt.write(f'{os.path.join("val", category, file)} {category}\n')

    with open(os.path.join(target_dir, 'meta', 'test.txt'), 'w') as test_txt:
        for category in categories:
            test_files = os.listdir(os.path.join(target_dir, 'test', category))
            for file in test_files:
                test_txt.write(f'{os.path.join("test", category, file)} {category}\n')

    print("数据集划分完成！")

if __name__ == '__main__':
    # 源文件夹路径
    source_folder = 'D:\大三\大三上\软件课设\数据集\DDSM2\data0'
    # 目标文件夹路径
    target_folder = 'D:\大三\大三上\软件课设\数据集\DDSM2\data'
    # 移动图片
    move_images(source_folder, target_folder)
    # 定义原始数据文件夹和目标数据集文件夹
    data_dir = 'data'
    target_dir = 'datasets'
    # 定义数据集划分比例
    train_split_ratio = 0.6
    val_split_ratio = 0.2
    test_split_ratio = 0.2
    # 创建数据集
    create_imagenet_dataset(data_dir, target_dir, train_split_ratio, val_split_ratio, test_split_ratio)

