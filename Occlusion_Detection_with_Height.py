import numpy as np
import cv2
import os
import json
import glob
import time

t1 = int(round(time.time() * 1000))

main_path = "F:\\acuSim\\acudataset\\Files\\"
emit_path = f"{main_path}emit"
text_path = f"{main_path}text"
json_path = f"{main_path}json"

img_files = sorted(glob.glob(f"{emit_path}\\*.jpg"))
txt_files = glob.glob(os.path.join(text_path, '*_1_output.txt'), recursive=True)
total_files = len(txt_files)
order = ['_1_output.txt']
count = 0
AoPT = 0
AoP = 0

for suffix in order:
    for idx, txt_file_path in enumerate(txt_files, 1):
        print(f"Processing: {txt_file_path} ({idx}/{total_files})")

        file_name_1 = os.path.basename(txt_file_path)
        model_name = txt_file_path.split("\\")[5].split("_1_output.txt")[0]
        # print(model_name)

        if file_name_1.endswith(suffix):

            with open(f"{text_path}\\{model_name}_1_output.txt", encoding="utf-8") as f:
                lines = f.readlines()

            len = 2048
            frame_num = 1
            output_lines = []
            output_data = {}

            for line in lines:

                if "frame" in line:
                    n = 0
                    frame_num = int(line.split("_")[1].split(":")[0])
                    new_line = "{:04d}".format(frame_num) + "\n"
                    output_lines.append(new_line)
                    output_img = np.zeros((len, len, 3), dtype=np.uint8)
                    frame_num_str = "{:04d}".format(frame_num)
                    output_data = {"label": []}

                else:
                    AoPT += 1
                    n += 1
                    bbox = []
                    name = line.split(",")[0]
                    for i in range(4):
                        bbox.append(float(line.split(",")[i + 1]))
                    
                    #depth
                    _D = float(line.split(",")[5])
                    _C = float(line.split(",")[6])
                    # print(f"\n {_C}")

                    scale_factor = 5.55555555
                    center = np.array([0.5, 0.5])

                    A = np.array([
                        [scale_factor, 0, center[0] * (1 - scale_factor)],
                        [0, scale_factor, center[1] * (1 - scale_factor)],
                        [0, 0, 1]])
                    B = np.array([0, 0, 1])

                    m_homogeneous = np.array([bbox[0], bbox[2], 1]).reshape(3, 1)
                    n_homogeneous = np.array([bbox[1], bbox[3], 1]).reshape(3, 1)

                    transformed_m_homogeneous = np.dot(A, m_homogeneous)
                    transformed_m = (transformed_m_homogeneous[:2, :] / transformed_m_homogeneous[2, :]).flatten()
                    transformed_n_homogeneous = np.dot(A, n_homogeneous)
                    transformed_n = (transformed_n_homogeneous[:2, :] / transformed_n_homogeneous[2, :]).flatten()

                    x_max = int(len - transformed_n[0] * len)
                    x_min = int(len - transformed_m[0] * len)
                    y_max = int(transformed_m[1] * len)
                    y_min = int(transformed_n[1] * len)

                    if n == 1:
                        img_temp = cv2.imread(img_files[count + frame_num - 1], cv2.IMREAD_GRAYSCALE)
                        _, img = cv2.threshold(img_temp, 127, 255, cv2.THRESH_BINARY)

                    region = img[y_min:y_max, x_min:x_max]

                    if cv2.countNonZero(region) < 20 or _C > 0:
                        # test_line_failed = "\033[31mFailed: frame {}: bbox of {}: {}, {}, {}, {}, number of pixels: {}\033[0m".format(frame_num, name, x_max, x_min, y_max, y_min, cv2.countNonZero(region))
                        # print(f"\033[31m\nNormal: {_C}\033[0m")
                        # print(test_line_failed)
                        continue

                    # test_line_success = "\033[32mCompleted: frame {}: bbox of {}: {}, {}, {}, {}, number of pixels: {}\033[0m".format(frame_num, name, x_max, x_min, y_max, y_min, cv2.countNonZero(region))
                    # print(f"\033[32m\nNormal: {_C}\033[0m")
                    # print(test_line_success)

                    AoP += 1
                    x_center = (int((x_min + x_max) / 2)) / len
                    y_center = (int((y_min + y_max) / 2)) / len

                    new_bbox = "{},{}".format(x_center, y_center)
                    new_line = "{},{},{}\n".format(name, new_bbox, _C)
                    output_lines.append(new_line)

                    label = {
                        "name": name,
                        "coordinate": {
                            "x": float(new_bbox.split(",")[0]),
                            "y": float(new_bbox.split(",")[1]),
                            "n": float(_C),
                            "h": float(_D/1000)
                        }
                    }
                    output_data["label"].append(label)

                # output to JSON
                json_output_path = os.path.join(json_path, f"{model_name}_{frame_num_str}.json")
                with open(json_output_path, "w", encoding="utf-8") as json_file:
                    json.dump(output_data, json_file, ensure_ascii=False)
                # print(f"\033[32mWritten to JSON file: {json_output_path}\033[0m")

            # output to txt
            with open(json_path + "\\" + model_name + ".txt", "w", encoding="utf-8") as f:
                f.writelines(output_lines)
            count += 144

t2 = int(round(time.time() * 1000))
print("Batch processing complete.")
print(f"{AoPT}, {AoP}")
print(t2 - t1)
