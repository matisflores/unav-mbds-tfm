import cv2

from mot.Grid import Grid

WINDOW_TITLE='ROI Selection'
ROI_COLOR = (0, 255, 0)
ROI_SELECTED_COLOR = (0, 200, 0)
ROI_THICKNESS = 1

class Roi:
    _grid: Grid = None
    _selecting = False
    _frame = None
    _start = None
    _end = None
    _roi_cells = None

    def __init__(self, grid: Grid):
        self._grid = grid
        self._roi_cells = []
    
    def define(self, frame):
        self._frame = frame

        # Create window and set mouse callback
        cv2.namedWindow(WINDOW_TITLE)
        cv2.setMouseCallback(WINDOW_TITLE, self._select_callback)
        cv2.imshow(WINDOW_TITLE, frame)
        cv2.waitKey(0)
        cv2.destroyWindow(WINDOW_TITLE)

    def _select_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._selecting = True
            self._start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._selecting:
            self._end = (x, y)
            self._select_plot()
        elif event == cv2.EVENT_LBUTTONUP:
            self._selecting = False
            self._end = (x, y)
            self._select_plot()
            self._select_zones()

    def _select_plot(self):
        frame = self._frame.copy()
        cv2.rectangle(frame, self._start, self._end, ROI_COLOR, ROI_THICKNESS)
        cv2.imshow(WINDOW_TITLE, frame)

    def _select_zones(self):
        # Extract selected zones and draw rectangles for each selected zone
        if self._start is not None and self._end is not None:
            x1, y1 = self._start
            x2, y2 = self._end
            cells = self._grid.cells
            cell_size = self._grid.cell_size

            selected = [cell for cell in cells if ((x1 <= cell[0] and cell[0] < x2) or (x1 <= cell[0] + cell_size and cell[0] + cell_size < x2)) and
                                                          ((y1 <= cell[1] and cell[1] < y2) or (y1 <= cell[1] + cell_size and cell[1] + cell_size < y2))]
            
            self._roi_cells = self._roi_cells + selected

        self.plot(self._frame, False)
        cv2.imshow(WINDOW_TITLE, self._frame)

    def plot(self, frame, copy=True):
        # Display the frame with selected zones
        frame = frame.copy() if copy else frame
        cell_size = self._grid.cell_size

        for cell in self._roi_cells:
            x, y, _, _, _ = cell
            cv2.rectangle(frame, (x, y), (x+cell_size, y+cell_size), ROI_SELECTED_COLOR, ROI_THICKNESS)
        
        return frame
    
    def in_zone(self, point: tuple) -> bool:
        """
        Check if a point is inside each selected zone.

        Args:
            point (tuple): The (x, y) coordinates of the point.

        Returns:
            bool: True if the point is inside at least one selected zone, False otherwise.
        """
        for zone in self._roi_cells:
            x, y, _, _, _ = zone
            cell_size = self._grid.cell_size
            cell_end_x = x + cell_size
            cell_end_y = y + cell_size
            
            if x <= point[0] <= cell_end_x and y <= point[1] <= cell_end_y:
                return True

        return False

    @property
    def selected_cells(self):
        return self._roi_cells