import cv2

class Video():
    _path: str = None
    _cap: cv2.VideoCapture = None

    def __init__(self, path: str):
        self._path = path

    def open(self):
        self._cap = cv2.VideoCapture(self._path)

        # Check if video opened successfully
        if not self._cap.isOpened():
            print("Error: Unable to open video.")
            exit()

        return self._cap
    
    def read(self, soft:bool = False):
        ret, frame = self._cap.read()
        if not ret and not soft:
            print("Error: Unable to read video.")
            exit()

        return frame
    
    def release(self):
        self._cap.release()
    
    @property
    def path(self):
        return self._path

    @property
    def cap(self):
        return self._cap

    @property
    def fps(self):
        return float(self._cap.get(cv2.CAP_PROP_FPS))