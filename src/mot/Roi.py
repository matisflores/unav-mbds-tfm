import cv2
import uuid

from utils.Grid import Grid

WINDOW_TITLE = 'ROI Selection'
ROI_COLOR = (0, 255, 0)
ROI_SELECTED_COLOR = (0, 200, 0)
ROI_THICKNESS = 1

class Roi:
    _id = None
    _grid: Grid = None
    _selecting = False
    _frame = None
    _start = None
    _end = None
    _roi_cells = None
    _count = 0

    def __init__(self, grid: Grid):
        self._id = str(uuid.uuid4())
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
        if frame is None:
            return None

        frame = frame.copy() if copy else frame

        cv2.rectangle(frame, self._start, self._end, ROI_SELECTED_COLOR, ROI_THICKNESS)
        cv2.putText(frame, str(self._count), (self._start[0], self._start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ROI_SELECTED_COLOR, ROI_THICKNESS, cv2.LINE_AA)
        '''
        cell_size = self._grid.cell_size

        for cell in self._roi_cells:
            x, y, _, _, _ = cell
            cv2.rectangle(frame, (x, y), (x+cell_size, y+cell_size), ROI_SELECTED_COLOR, ROI_THICKNESS)
        '''
        
        return frame
    
    def in_cell(self, point: tuple):
        for cell in self._roi_cells:
            x, y, _, _, _ = cell
            cell_end_x = x + self._grid.cell_size
            cell_end_y = y + self._grid.cell_size
            
            if x <= point[0] <= cell_end_x and y <= point[1] <= cell_end_y:
                return cell

        return None

    @property
    def selected_cells(self):
        return self._roi_cells