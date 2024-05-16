import motpy

from mot.Track import Track
from utils.Config import Config

def track_from_motpy(track: motpy.core.Track):
    return Track(track.id, track.box, track.score, track.class_id)

class MOTKalmanTracker:
    _tracker = None

    def __init__(self, fps):
        config = Config()
        
        self._tracker = motpy.tracker.MultiObjectTracker(
                dt=1 / fps,
                tracker_kwargs={'max_staleness': 5},
                model_spec={'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 2, 'q_var_pos': 5000., 'r_var_pos': 0.1},
                matching_fn_kwargs={'min_iou': float(config.kalman_min_iou), 'multi_match_min_iou': 0.93},
                active_tracks_kwargs={'min_steps_alive': int(config.kalman_min_steps_alive)}
            )
        
    def step(self, detections) -> list[Track]:
        tracks = self._tracker.step(detections=detections)
        return [track_from_motpy(track) for track in tracks]
    
    