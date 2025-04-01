import os
import cv2
import numpy as np
import glob
import time
import tensorflow as tf
from prettytable import PrettyTable


def list_tflite_devices():
    try:
        devices = tf.config.list_physical_devices()
        if not devices:
            print("Нет доступных устройств для TFLite.")
        else:
            for device in devices:
                print(f"Устройство: {device.device_type} - {device.name}")
    except Exception as e:
        print(f"Ошибка при получении устройств: {e}")


list_tflite_devices()
print(tf.config.list_physical_devices('GPU'))

def detect_image_with_model(modelpath, image, height, width):
    interpreter = tf.lite.Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    start_time = time.time()

    image_resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.float32 if input_dtype == np.float32 else np.uint8)

    if input_dtype == np.float32:
        input_data = (input_data - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    detected_objects = np.sum(scores > 0.6)
    end_time = time.time()

    return detected_objects, end_time - start_time


def tflite_detect_images(models, imgpath='TestImages'):
    images = glob.glob(os.path.join(imgpath, '*.*'))
    results = {name: [] for name in models.keys()}
    object_counts = {name: [] for name in models.keys()}

    table = PrettyTable(["Файл", *models.keys()])
    for image_path in images:
        image = cv2.imread(image_path)
        row = [os.path.basename(image_path)]

        for name, model in models.items():
            if model.endswith('.tflite'):
                interpreter = tf.lite.Interpreter(model_path=model)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                height, width = input_details[0]['shape'][1:3]
                objects, time_spent = detect_image_with_model(model, image, height, width)
            else:
                continue

            fps = 1 / time_spent if time_spent > 0 else 0
            row.append(f"{objects} ob, {fps:.2f} FPS")
            results[name].append([time_spent, fps])
            object_counts[name].append(objects)

        table.add_row(row)
    print(table)

    avg_results = {name: np.mean(vals, axis=0) for name, vals in results.items()}
    avg_objects = {name: np.mean(vals) for name, vals in object_counts.items()}

    best_fps_model = max(avg_results, key=lambda x: avg_results[x][1])
    best_objects_model = max(avg_objects, key=avg_objects.get)
    best_time_model = min(avg_results, key=lambda x: avg_results[x][0])

    comparison_table = PrettyTable(["Метрика", *models.keys(), "Лучший"])
    comparison_table.add_row(["Время (сек)", *[f"{avg_results[name][0]:.6f}" for name in models], best_time_model])
    comparison_table.add_row(["FPS", *[f"{avg_results[name][1]:.2f}" for name in models], best_fps_model])
    comparison_table.add_row(["Объекты", *[f"{avg_objects[name]:.2f}" for name in models], best_objects_model])

    print("\nСравнительная таблица:")
    print(comparison_table)


models = {
    "224x224": "detect.tflite",
    "quant 224x224": "detect_quant.tflite",
    #"320x320": "detect320.tflite",
    #"quant 320x320": "detect_quant320.tflite"
    "mn_car_224": "mn_car_224.tflite",
    "mn_car_224_quant": "mn_car_224_quant.tflite"
}

tflite_detect_images(models)
