import cv2

from config.config import Config

class Zones:
    _zones = None
    _zone_size = None
    _config = None

    def __init__(self):
        self._zones = []
        self._config = Config()
        self._zone_size = int(self._config.zone_size)

    def divide(self, frame):
        height, width, _ = frame.shape
        id = 0

        for y in range(0, height, self._zone_size):
            for x in range(0, width, self._zone_size):
                zone = frame[y:y+self._zone_size, x:x+self._zone_size]
                self._zones.append((x, y, zone, id, 0))
                id += 1

        print(f"Source: {height}x{width} - Zone Size: {self._zone_size}x{self._zone_size} - Zones: {len(self._zones)}")
        return self._zones
    
    def plot(self, frame):
        for x, y, zone, id, active in self._zones:
            cv2.rectangle(frame, (x, y), (x + self._zone_size, y + self._zone_size), (255,255,255), 1)
        return frame
    
    @property
    def zones(self):
        return self._zones

    @property
    def zone_size(self):
        return self._zone_size