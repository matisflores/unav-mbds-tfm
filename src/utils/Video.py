import cv2
import threading

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
    
    def read(self, soft:bool = False, downscale: float = 1., skip: int = 0):
        # Skip frames
        frame_nro = -1
        while frame_nro < skip:
            frame_nro += 1
            ret, frame = self._cap.read()
            if not ret and not soft:
                print("Error: Unable to read video.")
                exit()

        # Downscale frame
        if downscale != 1.:
            frame = cv2.resize(frame, fx=downscale, fy=downscale, dsize=None, interpolation=cv2.INTER_AREA)

        return frame
    
    def release(self):
        self._cap.release()

    def rewind(self):
        self.release()
        self.open()
    
    @property
    def path(self):
        return self._path

    @property
    def cap(self):
        return self._cap

    @property
    def fps(self):
        return float(self._cap.get(cv2.CAP_PROP_FPS))
    

class VideoProcessor():
    _thread: threading.Thread = None
    _stop: bool = False
    _read_frame: callable = None
    _on_frame: callable = None
    _step: int = 0

    def __init__(self, read_frame: callable, process_frame: callable):
        self._read_frame = read_frame
        self._on_frame = process_frame

    def __loop__(self):
        step = 0
        while True:
            frame = self._read_frame()

            self._on_frame(frame, step)
            step += 1

            if frame is None or self._stop:
                break

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self.__loop__)
            self._thread.start()

    def stop(self):
        self._stop = True
