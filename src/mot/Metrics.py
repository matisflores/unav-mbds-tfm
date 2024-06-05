import cv2
import numpy as np

from matplotlib import pyplot as plt
from collections import deque
from io import BytesIO

class Metric():
    _steps: deque = None
    _values: deque = None
    _name: str = None
    
    def __init__(self, name: str, maxlen: int = None):
        self._name = name
        self._steps = deque(maxlen=maxlen)
        self._values = deque(maxlen=maxlen)

    def store(self, value, step):
        self._steps.append(step)
        self._values.append(value)

    def each(self, action):
        for step, value in zip(self._steps, self._values):
            action(self._name, step, value)

    def plot(self, y_min = None, y_max = None):
        buffer = BytesIO()

        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
            plt.yticks(np.arange(y_min, y_max, 2))

        plt.plot(self._steps, self._values)
        plt.xlabel('Step')
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

    def stop(self, step):
        end = cv2.getTickFrequency() / (cv2.getTickCount() - self._start)
        self.store(end, step)

        return end
    
class MetricDetections(Metric):
    pass

class MetricTrackers(Metric):
    pass