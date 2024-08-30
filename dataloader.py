import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

length = 1026
epochs = 1000
learning_rate = 0.000075
batch_size = 64
target_size = (length, length)
input_shape = (length, length, 3)
max_keypoints = 174

log_dir = "logs/"
base_dir = "..."
map_dir = os.path.join(base_dir, "map.txt")
train_image_dir = os.path.join(base_dir, f"train\\image\\img_{length}")
train_json_dir = os.path.join(base_dir, "train\\label\\label\\")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
with open(map_dir, 'r', encoding='utf-8') as file:
    unique_acupoint_names = [line.strip() for line in file if line.strip()]
def create_category_encoding(names):
    categories = set(name[:2] for name in names)
    return {category: idx for idx, category in enumerate(sorted(categories))}
category_encoding = create_category_encoding(unique_acupoint_names)
num_categories = len(category_encoding)

def parse_json_and_map_weights(json_path, category_encoding):
    with open(json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    keypoint_info = data["label"]
    keypoints = np.full((max_keypoints, 3), [0, 0, length / 2])  # [0, 0, length / 2]
    weights = np.zeros(max_keypoints)  # 存储权重
    categories = np.zeros(max_keypoints, dtype=np.int32)

    for i, point in enumerate(keypoint_info):
        if i >= max_keypoints:
            break
        # 提取坐标
        x_p = int(float(point["coordinate"]["x"]) * length)
        y_p = int(float(point["coordinate"]["y"]) * length)
        h_p = int(float(point["coordinate"]["h"]) * (length / 0.8))

        keypoints[i] = [x_p, y_p, h_p]
        weights[i] = 0.8 + 0.2 * (-float(point["coordinate"]["n"]))

        category = point["name"][:2]
        if category in category_encoding:
            categories[i] = category_encoding[category]

    mask = create_mask(keypoints)
    return keypoints.flatten(), weights, categories, mask

def create_mask(keypoints):
    mask = np.any(keypoints != np.array([0, 0, length / 2]), axis=-1)
    mask = np.repeat(mask, 3)
    return mask.astype(np.float32)

def create_data_generator(image_dir, json_dir, batch_size, target_size, category_encoding):
    exclude_frames = list(range(12, 30)) + list(range(46, 108)) + list(range(119, 138))
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]

    filtered_image_paths = []
    for path in image_paths:
        frame_number = int(path.split('_')[-1].split('.')[0])
        if frame_number not in exclude_frames:
            filtered_image_paths.append(path)

    image_data_generator = tf.data.Dataset.from_tensor_slices(filtered_image_paths)
    image_data_generator = image_data_generator.map(
        lambda x: tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), channels=3), target_size) / 255.0,
        num_parallel_calls=tf.data.AUTOTUNE)

    json_files = sorted(os.listdir(json_dir))

    filtered_json_files = []
    for json_file in json_files:
        frame_number = int(json_file.split('_')[-1].split('.')[0])
        if frame_number not in exclude_frames:
            filtered_json_files.append(json_file)

    keypoints_array = []
    weights_array = []
    categories_array = []
    mask_array = []

    for json_file in filtered_json_files:
        json_path = os.path.join(json_dir, json_file)
        keypoints, weights, categories, mask = parse_json_and_map_weights(json_path, category_encoding)
        keypoints_array.append(keypoints)
        weights_array.append(weights)
        categories_array.append(categories)
        mask_array.append(mask)

    keypoints_tensor = tf.data.Dataset.from_tensor_slices(np.array(keypoints_array, dtype=np.float32))
    weights_tensor = tf.data.Dataset.from_tensor_slices(np.array(weights_array, dtype=np.float32))
    categories_tensor = tf.data.Dataset.from_tensor_slices(np.array(categories_array, dtype=np.int32))
    mask_tensor = tf.data.Dataset.from_tensor_slices(np.array(mask_array, dtype=np.float32))

    image_data_generator = image_data_generator.cache()
    data_generator = tf.data.Dataset.zip((image_data_generator, (keypoints_tensor, weights_tensor, categories_tensor, mask_tensor)))
    data_generator = data_generator.shuffle(buffer_size=1000)  # 调整缓冲区大小以适应数据集
    data_generator = data_generator.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # 预取以提高效率

    return data_generator

# 检查数据加载器
train_generator = create_data_generator(train_image_dir, train_json_dir, batch_size, target_size, category_encoding)
print(train_generator)

for images, (keypoints, weights, categories, mask) in train_generator.take(10):
    print(images.shape)
    print(keypoints.shape)
    print(weights.shape)
    print(categories.shape)
    print(mask.shape)
    break

