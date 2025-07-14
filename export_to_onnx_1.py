from onnxruntime_extensions.tools.pre_post_processing import *
from ultralytics import YOLO
import onnx


model_image_size = 640
model_iou_threshold = 0.4
model_score_threshold = 0.3  # 0.40  # 0.3
model_max_detections_per_class = 100
model_onnx_opset_version = 19
model_filename = '/Users/dmitrijpavlov/ics-it/ML/add_preprocessing_onnx/stand_prices_v10.onnx'
model_export_filename = '/Users/dmitrijpavlov/ics-it/ML/add_preprocessing_onnx/yolov11n_price_digits_v4.onnx'

# model = YOLO(model_filename)
# model.export(format='onnx')

# model = onnx.load(model_export_filename)

model = onnx.load(model_filename)
inputs = [create_named_value('image', onnx.TensorProto.UINT8, ['num_bytes'])]
image_size = model_image_size
num_classes = 10

pipeline = PrePostProcessor(inputs, model_onnx_opset_version)
pipeline.add_pre_processing(
    [
        ConvertImageToBGR(),
        Resize((image_size, image_size), policy='not_larger'),
        ChannelsLastToChannelsFirst(),
        ImageBytesToFloat(),
        Unsqueeze([0])
    ]
)
post_processing_steps = [
    Squeeze([0]),
    Transpose([1, 0]),
    Split(num_outputs=2, axis=-1, splits=[4, num_classes]),
    SelectBestBoundingBoxesByNMS(
        iou_threshold=model_iou_threshold,
        score_threshold=model_score_threshold,
        max_boxes_per_class=model_max_detections_per_class),
    (ScaleNMSBoundingBoxesAndKeyPoints(name='ScaleBoundingBoxes'),
     [
         utils.IoMapEntry('original_image', producer_idx=0, consumer_idx=0),
         utils.IoMapEntry('Resize', producer_idx=0, consumer_idx=1)
     ])
]
pipeline.add_post_processing(post_processing_steps)

modified_model = pipeline.run(model)
onnx.checker.check_model(modified_model)
onnx.save_model(modified_model, model_export_filename)
