import cv2

class Grid:
    _cells = None
    _cell_size = None

    def __init__(self, cell_size: int):
        self._cells = []
        self._cell_size = cell_size

    def divide(self, frame):
        height, width, _ = frame.shape
        id = 0

        for y in range(0, height, self._cell_size):
            for x in range(0, width, self._cell_size):
                frame_part = frame[y:y+self._cell_size, x:x+self._cell_size]
                self._cells.append((x, y, frame_part, id, 0))
                id += 1

        print(f"Source: {height}x{width} - Zone Size: {self._cell_size}x{self._cell_size} - Zones: {len(self._cells)}")
        return self._cells
    
    def plot(self, frame):
        for x, y, _, _, _ in self._cells:
            cv2.rectangle(frame, (x, y), (x + self._cell_size, y + self._cell_size), (255,255,255), 1)
        return frame
    
    @property
    def cells(self):
        return self._cells

    @property
    def cell_size(self):
        return self._cell_size