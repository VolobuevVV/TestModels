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


def load_pb_graph(modelpath):
    tf.config.optimizer.set_jit(True)
    with tf.io.gfile.GFile(modelpath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def detect_pb_model(image, sess):
    need_classes = [1]
    img_h, img_w, _ = image.shape
    start_time = time.time()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    inputs = np.reshape(img, (1, 224, 224, 3))

    out = sess.run(
        [
            sess.graph.get_tensor_by_name('num_detections:0'),
            sess.graph.get_tensor_by_name('detection_scores:0'),
            sess.graph.get_tensor_by_name('detection_boxes:0'),
            sess.graph.get_tensor_by_name('detection_classes:0')
        ],
        feed_dict={'image_tensor:0': inputs}
    )

    num_detections = int(out[0][0])
    boxes = out[2][0][:num_detections]
    scores = out[1][0][:num_detections]
    class_ids = out[3][0][:num_detections]

    valid_detections = scores > 0.8

    end_time = time.time()

    valid_class_ids = np.array([cid in need_classes for cid in class_ids], dtype=bool)
    valid_combined = valid_detections & valid_class_ids
    filtered_class_ids = class_ids[valid_combined]

    return len(filtered_class_ids), end_time - start_time


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
    pb_graph = load_pb_graph('frozen_inference_graph_old.pb')
    with tf.compat.v1.Session(graph=pb_graph) as sess:
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
                    objects, time_spent = detect_pb_model(image, sess)

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
    "320x320": "detect320.tflite",
    "quant 320x320": "detect_quant320.tflite",
    "old model 224x224": "frozen_inference_graph_old.pb"
}

tflite_detect_images(models)
