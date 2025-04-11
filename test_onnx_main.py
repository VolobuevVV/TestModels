import onnxruntime as ort
import numpy as np
import cv2
import os
import glob
import time
from prettytable import PrettyTable

model_path = 'model512.onnx'
image_folder = 'TestImages'

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

images = glob.glob(os.path.join(image_folder, '*.*'))
results = []
table = PrettyTable(["Файл", "Время (сек)", "FPS", "Объектов"])

for image_path in images:
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_np, (320, 320))

    input_tensor = np.expand_dims(image_resized, axis=0).astype(np.uint8)

    start_time = time.time()

    outputs = session.run(output_names, {input_name: input_tensor})

    detection_scores = outputs[4]
    num_objects = np.sum(detection_scores[0] > 0.5)

    end_time = time.time()
    time_spent = end_time - start_time
    fps = 1 / time_spent if time_spent > 0 else 0

    table.add_row([os.path.basename(image_path), f"{time_spent:.4f}", f"{fps:.2f}", int(num_objects)])
    results.append([time_spent, fps, num_objects])

print(table)

avg_time = np.mean([r[0] for r in results])
avg_fps = np.mean([r[1] for r in results])
avg_objects = np.mean([r[2] for r in results])

summary_table = PrettyTable(["Метрика", "Среднее значение"])
summary_table.add_row(["Время (сек)", f"{avg_time:.4f}"])
summary_table.add_row(["FPS", f"{avg_fps:.2f}"])
summary_table.add_row(["Объектов", f"{avg_objects:.2f}"])

print("\nСредние значения:")
print(summary_table)
