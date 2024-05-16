from mot.Roi import Roi
from mot.Track import Track

class Event_StepEnd():
    frame = None

    def __init__(self, frame):
        self.frame = frame

class Event_TrackerInRoi():
    track: Track
    roi: Roi

    def __init__(self, track: Track, roi: Roi):
        self.track: Track = track
        self.roi: Roi = roi

    def save(self):
        if self.track is not None and self.roi is not None:
            print(self.track._id)
