import cv2

from mot.Zones import Zones

WINDOW_TITLE='ROI Selection'
ROI_COLOR = (0, 255, 0)
ROI_SELECTED_COLOR = (0, 200, 0)
ROI_THICKNESS = 1

class Roi:
    _zones: Zones = None
    _selecting = False
    _frame = None
    _start = None
    _end = None
    _roi_zones = None

    def __init__(self, zones: Zones):
        self._zones = zones
        self._roi_zones = []
    
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
            zones = self._zones.zones
            zone_size = self._zones.zone_size

            selected = [zone for zone in zones if ((x1 <= zone[0] and zone[0] < x2) or (x1 <= zone[0] + zone_size and zone[0] + zone_size < x2)) and
                                                          ((y1 <= zone[1] and zone[1] < y2) or (y1 <= zone[1] + zone_size and zone[1] + zone_size < y2))]
            
            self._roi_zones = self._roi_zones + selected

        self.plot(self._frame, False)
        cv2.imshow(WINDOW_TITLE, self._frame)

    def plot(self, frame, copy=True):
        # Display the frame with selected zones
        frame = frame.copy() if copy else frame
        zone_size = self._zones.zone_size

        for zone in self._roi_zones:
            x, y, _, _, _ = zone
            cv2.rectangle(frame, (x, y), (x+zone_size, y+zone_size), ROI_SELECTED_COLOR, ROI_THICKNESS)
        
        return frame
    
    def in_zone(self, point: tuple) -> bool:
        """
        Check if a point is inside each selected zone.

        Args:
            point (tuple): The (x, y) coordinates of the point.

        Returns:
            bool: True if the point is inside at least one selected zone, False otherwise.
        """
        for zone in self._roi_zones:
            x, y, _, _, _ = zone
            zone_size = self._zones.zone_size
            zone_end_x = x + zone_size
            zone_end_y = y + zone_size
            
            if x <= point[0] <= zone_end_x and y <= point[1] <= zone_end_y:
                return True

        return False

    @property
    def selected_zones(self):
        return self._roi_zones