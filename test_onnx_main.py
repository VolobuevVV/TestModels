import onnxruntime as ort
import numpy as np
import cv2
import os
import glob
import time

model_path = r"C:\Users\vladi\Downloads\best(1).onnx"
image_folder = 'images'

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

images = glob.glob(os.path.join(image_folder, '*.*'))
results = []

for image_path in images:
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_np, (224, 224))

    input_tensor = np.expand_dims(image_resized, axis=0).astype(np.float32)
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

    start_time = time.time()

    outputs = session.run(output_names, {input_name: input_tensor})
    predictions = np.squeeze(outputs).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > 0.2, :]
    scores = scores[scores > 0.5]
    print(predictions)
    end_time = time.time()
    time_spent = end_time - start_time
    fps = 1 / time_spent if time_spent > 0 else 0
    results.append([time_spent, fps])


avg_time = np.mean([r[0] for r in results])
avg_fps = np.mean([r[1] for r in results])

print("Время (сек)", f"{avg_time:.4f}")
print("FPS", f"{avg_fps:.2f}")

