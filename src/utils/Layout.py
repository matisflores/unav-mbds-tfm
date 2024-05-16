import cv2
import tkinter as tk
import numpy as np

SCREEN_OFFSET_Y = 70
SCREEN_MENU_BAR = 0.3

class Screen:
    name: str = ''
    width: int = 0
    height: int = 0
    offset_x: int = 0
    offset_y: int = 0
    menu_bar: int = 0
    resize: bool = False

    def __init__(self, name: str, width: float = 0, height: float = 0, offset_x: float = 0, offset_y: float = 0, resize: bool = False, show: bool = True):
        screen = tk.Tk()
        max_width = screen.winfo_screenwidth()
        max_height = screen.winfo_screenheight()
        aspect = max_height/max_width

        self.name = name
        self.resize = resize
        self.offset_x = int(max_width * float(offset_x))
        self.offset_y = int(max_height * float(offset_y)) - SCREEN_OFFSET_Y

        if width > 0 and width <= 1:
            self.width = int(max_width * float(width))
        elif width > 1 and width <= max_width:
            self.width = int(width)
        else:
            self.width = max_width

        if height > 0 and height <= 1:
            self.height = int(max_height * float(height))
        elif height > 1 and height <= max_height:
            self.height = int(height)
        else:
            self.height = int(aspect * self.width)

        if offset_y > 0:
            self.menu_bar = int(max_height * SCREEN_MENU_BAR)

        if show:
            self.show(np.zeros((self.height, self.width, 3), dtype=np.uint8))

    @property
    def position(self):
        return (self.offset_x, self.offset_y)
    
    def show(self, frame, delay: int = 1):
        if self.resize:
            frame = cv2.resize(frame, (self.width, self.height))

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.name, frame)
        cv2.moveWindow(self.name, self.offset_x, self.offset_y + self.menu_bar)

        return cv2.waitKey(delay)

class Layout:
    _screens: dict[str, Screen] = {}

    def __init__(self, screens: dict[str, Screen] = {}):
        self._screens = screens

    def show(self, screen: str, frame, delay: int = 1):
        scr = self._screens.get(screen, None)

        if scr is not None:
            return scr.show(frame, delay)
        
        return None
    
    def close(self):
        cv2.destroyAllWindows()
