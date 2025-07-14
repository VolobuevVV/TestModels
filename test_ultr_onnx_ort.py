import logging
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import onnxruntime_extensions
logger = logging.getLogger()

providers = ['CPUExecutionProvider']
session_options = ort.SessionOptions()
session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
model_onnx = ort.InferenceSession("models/best(1)_preprocessed.onnx", sess_options=session_options, providers=providers)

images_folder = "images"

image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

total_time = 0.0
num_images = len(image_files)

for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image_to_draw = cv2.imread(image_path)

        start_time = time.time()
        output = model_onnx.run(None, {'image': image})[0]
        output2 = np.squeeze(output).T

        end_time = time.time()
        execution_time = end_time - start_time
        total_time += execution_time

        """
        for i in range(output2.shape[1]):
            x1, y1, w, h, conf, cls = output2[:, i]
            x2 = x1 + w
            y2 = y1 + h
            print(x1, y1, x2, y2, conf, cls)
            if conf > 0.2:
                cv2.rectangle(image_to_draw, (int(x1 - ((x2 - x1) / 2)), int(y1 - ((y2 - y1) / 2))), (int(x2 - ((x2 - x1) / 2)), int(y2 - ((y2 - y1) / 2))), (0, 255, 0), 2)

        cv2.imshow('Detections', image_to_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
if num_images > 0:
    average_time = total_time / num_images
    logger.log(logging.WARNING, f"Среднее время выполнения для {num_images} изображений: {average_time:.6f} секунд, FPS - {1 / average_time}")