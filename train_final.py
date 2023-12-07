from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8s.yaml')  # build a new model from YAML
model = YOLO('/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/v8/yolov8_swinTrans.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
if __name__ == '__main__':
    model.train(data='/root/autodl-tmp/ultralytics-main/data/data.yaml', pretrained='/root/autodl-tmp/ultralytics-main/yolov8n.pt', epochs=400, imgsz=640, device='0')

