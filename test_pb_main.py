import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import time
from prettytable import PrettyTable
tf.compat.v1.disable_eager_execution()  # Отключить Eager Execution

model_path = 'saved_model'
image_folder = 'TestImages'

loaded_model = tf.saved_model.load(model_path)
infer = loaded_model.signatures['serving_default']

images = glob.glob(os.path.join(image_folder, '*.*'))
results = []
table = PrettyTable(["Файл", "Время (сек)", "FPS", "Объектов"])

for image_path in images:
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_np, (320, 320))

    input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    start_time = time.time()
    output_dict = infer(input_tensor)
    end_time = time.time()

    detection_scores = output_dict['detection_scores'].numpy()
    num_objects = np.sum(detection_scores[0] > 0.5)
    time_spent = end_time - start_time
    fps = 1 / time_spent if time_spent > 0 else 0

    table.add_row([os.path.basename(image_path), f"{time_spent:.4f}", f"{fps:.2f}", int(num_objects)])
    results.append([time_spent, fps, num_objects])

print(table)

# Подсчёт средних значений
avg_time = np.mean([r[0] for r in results])
avg_fps = np.mean([r[1] for r in results])
avg_objects = np.mean([r[2] for r in results])

summary_table = PrettyTable(["Метрика", "Среднее значение"])
summary_table.add_row(["Время (сек)", f"{avg_time:.4f}"])
summary_table.add_row(["FPS", f"{avg_fps:.2f}"])
summary_table.add_row(["Объектов", f"{avg_objects:.2f}"])

print("\nСредние значения:")
print(summary_table)

