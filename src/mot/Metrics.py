from collections import deque
from io import BytesIO
import cv2
from matplotlib import pyplot as plt
import numpy as np


class Metric():
    _history: deque = None
    _name: str = None
    
    def __init__(self, name: str, maxlen: int):
        self._name = name
        self._history = deque(maxlen=maxlen)

    def store(self, value):
        self._history.append(value)

    def plot(self):
        buffer = BytesIO()
        plt.plot(list(range(len(self._history))), self._history, color='blue')
        plt.xlabel('Steps')
        plt.ylabel(self._name)
        plt.savefig(buffer, format='png')
        plt.clf()
        buffer.seek(0)
        time_series_img = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        return cv2.imdecode(time_series_img, cv2.IMREAD_COLOR)
        

class MetricFPS(Metric):
    _start: int = None

    def start(self):
        self._start = cv2.getTickCount()

    def stop(self):
        end = cv2.getTickFrequency() / (cv2.getTickCount() - self._start)
        self.store(end)

        return end
    
class MetricDetections(Metric):
    pass

class MetricTrackers(Metric):
    pass