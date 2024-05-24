import threading

class Algorithm():
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
