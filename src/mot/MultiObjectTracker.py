import motpy
import numpy as np

from typing import Dict

from mot.Track import Track
from mot.Trackers import GHTracker, KalmanTracker, ParticleTracker
from utils.Config import Config

def track_from_motpy(track: motpy.core.Track):
    if track is None:
        return None
    
    return Track(track.id, track.box, track.score, track.class_id)

def previous_to_dict(previous_tracks):
    return {track.id: track for track in previous_tracks}

class _MultiObjectTracker(motpy.tracker.MultiObjectTracker):
    def __init__(self,
                 dt: float,
                 tracker_clss = None,
                 model_spec: Dict = {},
                 matching_fn: motpy.tracker.BaseMatchingFunction = None,
                 tracker_kwargs: Dict = None,
                 matching_fn_kwargs: Dict = None,
                 active_tracks_kwargs: Dict = None) -> None:
        
        super().__init__(dt, model_spec, matching_fn, tracker_kwargs, matching_fn_kwargs, active_tracks_kwargs)

        self.tracker_clss = tracker_clss
        self.tracker_kwargs['model_kwargs'] = model_spec
        self.tracker_kwargs['model_kwargs']['dt'] = dt

class MultiObjectTracker():
    _tracker = None
    _active_tracks = []
    _previous_tracks = []

    def __init__(self, tracker: _MultiObjectTracker):
        self._tracker = tracker
        
    def update_tracks(self, active_tracks):
        self._previous_tracks = self._active_tracks
        self._active_tracks = active_tracks

        return self._active_tracks, (len(self._active_tracks) - len(self._previous_tracks))

    def step(self, detections) -> list[motpy.core.Track]:
        if len(detections) > 0:
            return self.update_tracks(self._tracker.step(detections=detections))
        
        # predict state in all trackers
        for t in self._tracker.trackers:
            t.predict()

        return self.update_tracks(self._tracker.active_tracks(**self._tracker.active_tracks_kwargs))
    
    def error(self):
        errors = []

        for t in self._tracker.trackers:
            errors.append(t.error()[1])

        errors = [e for e in errors if e is not None]

        if len(errors) > 0:
            return (np.min(errors), np.mean(errors), np.max(errors), np.mean([e**2 for e in errors]))
        
        return None

    @staticmethod
    def make(type: str, fps: int):
        type = type.upper()
        config = Config()

        if type == 'KALMAN':
            return MultiObjectTracker(_MultiObjectTracker(
                dt=1 / fps,
                tracker_clss=KalmanTracker,
                tracker_kwargs={'max_staleness': 5},
                model_spec={'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 2, 'q_var_pos': 5000., 'r_var_pos': 0.1},
                matching_fn_kwargs={'min_iou': config.get('min_iou', float), 'multi_match_min_iou': 0.93},
                active_tracks_kwargs={'min_steps_alive': config.get('detection_rate', int) + 1}
            ))
        elif type == 'GH':
            return MultiObjectTracker(_MultiObjectTracker(
                dt=1 / fps,
                tracker_clss=GHTracker,
                tracker_kwargs={'max_staleness': 5},
                model_spec={},
                matching_fn_kwargs={'min_iou': config.get('min_iou', float), 'multi_match_min_iou': 0.93},
                active_tracks_kwargs={'min_steps_alive': config.get('detection_rate', int) + 1}
            ))
        elif type == 'PARTICLE':
            return MultiObjectTracker(_MultiObjectTracker(
                dt=1 / fps,
                tracker_clss=ParticleTracker,
                tracker_kwargs={'max_staleness': 5},
                model_spec={},
                matching_fn_kwargs={'min_iou': config.get('min_iou', float), 'multi_match_min_iou': 0.93},
                active_tracks_kwargs={'min_steps_alive': config.get('detection_rate', int) + 1}
            ))            
