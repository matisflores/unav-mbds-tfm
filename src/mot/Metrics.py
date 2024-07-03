import cv2
import numpy as np

from matplotlib import pyplot as plt
from collections import deque
from io import BytesIO

from utils.DB import DB
from utils.Grid import Grid

def frame_cell_scores(db_file: str, frame, grid: Grid, color):
    db = DB(db_file)
    cell_scores = db.load_cell_scores()
    cell_scores = {cell[0]: cell[1] for cell in cell_scores}
    grid._cells = [(cell[0],cell[1],cell[2],cell[3],cell_scores.get(cell[3],0)) for cell in grid._cells]

    blue = np.full((grid.cell_size,grid.cell_size,3), color, np.uint8)
    frame_scores = frame.copy()
    for x, y, img, _, score in grid._cells:
        frame_scores[y:y+grid.cell_size, x:x+grid.cell_size] = cv2.addWeighted(img, 1, blue, score, 0.0)

    return frame_scores

def frame_cell_traffic(db_file, frame, grid: Grid, color):
    db = DB(db_file)
    cell_qty = db.load_cell_qty()
    cell_qty = {cell[0]: cell[1] for cell in cell_qty}

    scale = 1.0/max(cell_qty.values())
    for k in cell_qty:
        cell_qty[k] = cell_qty[k] * scale
    
    grid._cells = [(cell[0],cell[1],cell[2],cell[3],cell_qty.get(cell[3],0)) for cell in grid._cells]

    green = np.full((grid.cell_size,grid.cell_size,3), color, np.uint8)
    frame_qty = frame.copy()
    for x, y, img, _, score in grid._cells:
        frame_qty[y:y+grid.cell_size, x:x+grid.cell_size] = cv2.addWeighted(img, 1, green, score, 0.0)

    return frame_qty

def stats_tracking_duration(db_file):
    def timestamp_diff_to_sec(timestamp_difference):
        return int(cv2.getTickFrequency() / timestamp_difference) if timestamp_difference != 0 else 0


    buffer = BytesIO()
    db = DB(db_file)
    results = db.load_tracking_duration()

    results = [(row[0], row[1], row[2], timestamp_diff_to_sec(row[3])) for row in results]

    '''
    # Print the formatted results
    for row in results:
        tracker, min_timestamp, max_timestamp, timestamp_difference = row
        print(f"Tracker: {tracker}")
        print(f"Minimum Timestamp: {min_timestamp}")
        print(f"Maximum Timestamp: {max_timestamp}")
        print(f"Timestamp Difference (seconds): {timestamp_difference}")
        print()
    '''

    # Fetch the timestamp differences
    timestamp_differences = [row[3] for row in results]

    # Plot the distribution
    #plt.hist(timestamp_differences, color='skyblue', edgecolor='black')
    plt.boxplot(timestamp_differences)
    plt.title('Distribution of Trackers duration')
    #plt.xlabel('Timestamp Difference (seconds)')
    #plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(buffer, format='png')
    plt.clf()
    buffer.seek(0)
    time_series_img = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
    return cv2.imdecode(time_series_img, cv2.IMREAD_COLOR)

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

    def calculate_moving_average(self, values, window_size):
        values_array = np.array(values)
        moving_avg = np.convolve(values_array, np.ones(window_size) / window_size, mode='valid')
        return moving_avg

    def plot(self, y_min = None, y_max = None, window_size = 30):
        buffer = BytesIO()

        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
            plt.yticks(np.arange(y_min, y_max, 2))

        plt.plot(self._steps, self._values, label = 'Original')

        # Calculate and plot the moving average
        if len(self._values) >= window_size:
            steps_list = list(self._steps)
            values_list = list(self._values)
            moving_avg = self.calculate_moving_average(values_list, window_size)
            moving_avg_steps = steps_list[window_size - 1:] 
            plt.plot(moving_avg_steps, moving_avg, label = 'Moving Average')

        plt.xlabel('Frame')
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
        end = int(cv2.getTickFrequency() / (cv2.getTickCount() - self._start))
        self.store(end, step)

        return end
    
class MetricDetections(Metric):
    pass

class MetricTrackers(Metric):
    pass

class MetricTrackerErrors(Metric):
    pass

class MetricTrackersDelta(Metric):
    pass