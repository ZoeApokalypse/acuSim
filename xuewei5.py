from PIL import Image, ImageDraw, ImageFont
import numpy as np

color1 = (255, 0, 20)
color2 = (255, 120, 20)
cyan = (0, 255, 255)
dark_blue = (0, 0, 200)
green = (0, 255, 0)

name = "asher"
length = 1024

def choose_color(n):
    if n < 0:
        t = (n + 1) / 0.5
        return (int((1 - t) * color1[0] + t * color2[0]),
                int((1 - t) * color1[1] + t * color2[1]),
                int((1 - t) * color1[2] + t * color2[2]),
                255)
    else:
        t = n / 0.5
        return (int((1 - t) * cyan[0] + t * dark_blue[0]),
                int((1 - t) * cyan[1] + t * dark_blue[1]),
                int((1 - t) * cyan[2] + t * dark_blue[2]),
                180)

# 读取txt文件
txt_file_path = f"{name}_filtered.txt"
labels = []
with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
    for line in txt_file:
        parts = line.strip().split(',')
        if len(parts) == 7:  # 确保每行都有正确的数据格式
            labels.append({
                'name': parts[0],
                'x_min': float(parts[1]),
                'x_max': float(parts[2]),
                'y_min': float(parts[3]),
                'y_max': float(parts[4]),
                'n': float(parts[6])
            })

image_path = f"{name}_c.png"
image = Image.open(image_path).convert("RGBA")
draw = ImageDraw.Draw(image, "RGBA")

# 加载黑体字体文件，并设置字体大小
font_path = "simhei.ttf"  # 请确保路径正确
font_size = 16
font = ImageFont.truetype(font_path, font_size)

for label in labels:
    if label['n'] >= 0:
        continue

    color = choose_color(label['n'])
    scale_factor = 5.55555555
    center = np.array([0.5, 0.5])

    A = np.array([
        [scale_factor, 0, center[0] * (1 - scale_factor)],
        [0, scale_factor, center[1] * (1 - scale_factor)],
        [0, 0, 1]])
    B = np.array([0, 0, 1])

    m_homogeneous = np.array([label['x_min'], label['y_min'], 1]).reshape(3, 1)
    n_homogeneous = np.array([label['x_max'], label['y_max'], 1]).reshape(3, 1)

    transformed_m_homogeneous = np.dot(A, m_homogeneous)
    transformed_m = (transformed_m_homogeneous[:2, :] / transformed_m_homogeneous[2, :]).flatten()
    transformed_n_homogeneous = np.dot(A, n_homogeneous)
    transformed_n = (transformed_n_homogeneous[:2, :] / transformed_n_homogeneous[2, :]).flatten()

    offset = 1

    x_max = int(length - transformed_n[0] * length) + offset
    x_min = int(length - transformed_m[0] * length) - offset
    y_max = int(transformed_m[1] * length) + offset
    y_min = int(transformed_n[1] * length) - offset
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    draw.ellipse([(center_x-3, center_y-3), (center_x+3, center_y+3)], fill=green, outline=green)
    acupoint_name = label['name'].split('_')[0]
    text_bbox = draw.textbbox((0, 0), acupoint_name, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = x_min
    text_y = y_max + 2
    draw.text((text_x, text_y), acupoint_name, fill=color, font=font)

image.save(f"{name}_annotated_image_new.png")
image.show()