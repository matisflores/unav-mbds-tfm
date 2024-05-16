import threading

from utils.Video import Video

class Algorithm():
    _thread: threading.Thread = None
    _video: Video = None
    _stop: bool = False
    _on_frame: callable = None
    _step: int = 0

    def __init__(self, video: Video, process_frame: callable):
        self._video = video
        self._on_frame = process_frame

    def __loop__(self):
        step = 0
        while True:
            frame = self._video.read(soft=True)

            if frame is None or self._stop:
                break

            self._on_frame(frame, step)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self.__loop__)
            self._thread.start()

    def stop(self):
        self._stop = True
