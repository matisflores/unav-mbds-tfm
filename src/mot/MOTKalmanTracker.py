import motpy

from mot.Track import Track
from utils.Config import Config

def track_from_motpy(track: motpy.core.Track):
    if track is None:
        return None
    
    return Track(track.id, track.box, track.score, track.class_id)

def previous_to_dict(previous_tracks):
    return {track.id: track for track in previous_tracks}

class MOTKalmanTracker:
    _tracker = None
    _active_tracks = []
    _previous_tracks = []

    def __init__(self, fps):
        config = Config()
        
        self._tracker = motpy.tracker.MultiObjectTracker(
                dt=1 / fps,
                tracker_kwargs={'max_staleness': 5},
                model_spec={'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 2, 'q_var_pos': 5000., 'r_var_pos': 0.1},
                matching_fn_kwargs={'min_iou': float(config.kalman_min_iou), 'multi_match_min_iou': 0.93},
                active_tracks_kwargs={'min_steps_alive': int(config.kalman_min_steps_alive)}
            )
        
    def update_tracks(self, active_tracks):
        self._previous_tracks = self._active_tracks
        self._active_tracks = active_tracks

        return self._active_tracks

    def step(self, detections) -> list[motpy.core.Track]:
        if len(detections) > 0:
            return self.update_tracks(self._tracker.step(detections=detections))
        
        # predict state in all trackers
        for t in self._tracker.trackers:
            t.predict()

        return self.update_tracks(self._tracker.active_tracks(**self._tracker.active_tracks_kwargs))

    