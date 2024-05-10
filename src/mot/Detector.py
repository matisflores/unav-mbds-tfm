from config.config import Config
from motpy.core import Detection
from ultralytics import YOLO

PERSON=0

class YOLODetector:
    _detector = None
    _confidence_threshold = None

    def __init__(self):
        config = Config()
        self._detector = YOLO(config.path_assets_dir + '/yolov8n.pt')
        self._confidence_threshold = float(config.yolo_confidence_threshold)

    def detect(self, frame) -> list[Detection]:
        results = self._detector(frame, verbose=False)
        detections = [Detection(box=b, score=s, class_id=l) for b, s, l in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), results[0].boxes.cls.cpu().numpy().astype(int))]
        return [i for i in detections if i.class_id == PERSON and i.score >= self._confidence_threshold]