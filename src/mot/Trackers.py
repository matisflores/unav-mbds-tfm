import numpy as np

from filterpy.gh import GHFilter
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.monte_carlo import systematic_resample
from motpy.core import Box, Detection, Vector, setup_logger
from motpy.model import Model
from motpy.tracker import SingleObjectTracker
from numpy.random import randn
from typing import Optional

import scipy

logger = setup_logger(__name__)

class Tracker(SingleObjectTracker):
    _center = None
    _width = None
    _height = None

    _error_min = None
    _error_max = None
    _error_mean = None
    _error_var = None

    def __init__(self, box: Box, **kwargs):
        super(Tracker, self).__init__(**kwargs)

        self._center = self._box_to_point(box)
        self._width = box[2] - box[0]
        self._height = box[3] - box[1]

    def _calc_error_distribution(self, error):
        self._error_min = np.min(error)
        self._error_max = np.mean(error)
        self._error_mean = np.max(error)
        self._error_var = np.mean(error**2)

    def error(self):
        #(min, mean, max, var)
        return (self._error_min, self._error_mean, self._error_max, self._error_var)
    
    def _box_to_point(self, box: Box):
        #[xmin, ymin, xmax, ymax]
        return np.array((int((box[0] + box[2])/2), int((box[1] + box[3])/2)))
    
    def box(self) -> Box:
        w = self._width / 2
        h = self._height / 2
        return np.array([self._center[0] - w, self._center[1] - h, self._center[0] + w, self._center[1] + h])
    
    def is_invalid(self) -> bool:
        try:
            has_nans = any(np.isnan(self._center))
            return has_nans
        except Exception as e:
            logger.warning(f'invalid tracker - exception: {e}')
            return True

class GHTracker(Tracker):
    """ A single object tracker using GH filter """

    def __init__(self,
                 model_kwargs: dict = { 'dt': 1 },
                 x0: Optional[Vector] = None,
                 box0: Optional[Box] = None,
                 **kwargs) -> None:

        super(GHTracker, self).__init__(box0, **kwargs)

        self._tracker: GHFilter = GHFilter(x=self._center, g=0.01, h=0.1, dx=0, dt=model_kwargs['dt'])

    def _predict(self) -> None:
        self._tracker.dx = self._tracker.dx_prediction
        self._tracker.x  = self._tracker.x_prediction

    def _update_box(self, detection: Detection) -> None:
        x = self._box_to_point(detection.box)

        error = self._tracker.x - x
        self._calc_error_distribution(error)

        self._tracker.update(x)
        self._center = x

class KalmanTracker(Tracker):
    """ A single object tracker using Kalman filter with specified motion model specification """

    def __init__(self,
                 model_kwargs: dict = { 'dt': 1 },
                 x0: Optional[Vector] = None,
                 box0: Optional[Box] = None,
                 **kwargs) -> None:

        super(KalmanTracker, self).__init__(box0, **kwargs)

        self.model_kwargs: dict = model_kwargs
        self.model = Model(**self.model_kwargs)

        tracker = KalmanFilter(dim_x=self.model.state_length, dim_z=self.model.measurement_length)
        tracker.F = self.model.build_F()
        tracker.Q = self.model.build_Q()
        tracker.H = self.model.build_H()
        tracker.R = self.model.build_R()
        tracker.P = self.model.build_P()
        tracker.x = self.model.box_to_x(box0)

        self._tracker: KalmanFilter = tracker

    def _predict(self) -> None:
        self._tracker.predict()
        self._center = self._box_to_point(self.model.x_to_box(self._tracker.x))

    def _update_box(self, detection: Detection) -> None:
        error = self._tracker.x - self.model.box_to_x(detection.box)
        self._calc_error_distribution(error)

        z = self.model.box_to_z(detection.box)
        self._tracker.update(z)
        self._center = self._box_to_point(detection.box)

class ParticleTracker(Tracker):
    """ A single object tracker using Particle filter """

    _particles_qty = None
    _particles = None
    _weights = None
    _dt = 1.

    def __init__(self,
                 model_kwargs: dict = { 'dt': 1 },
                 x0: Optional[Vector] = None,
                 box0: Optional[Box] = None,
                 **kwargs) -> None:

        super(ParticleTracker, self).__init__(box0, **kwargs)

        self._particles_qty = 500
        self._dt = model_kwargs['dt']
        std = (5, 5, np.pi/4)

        self._particles = self._create_gaussian_particles(mean=(self._center[0], self._center[1], np.pi/4), std=std, N=self._particles_qty)
        self._weights = np.ones(self._particles_qty) / self._particles_qty

    def _predict(self) -> None:
        u = (0., 1.)
        std = (.2, .05)

        # update heading (curso, direccion)
        self._particles[:, 2] += u[0] + (randn(self._particles_qty) * std[0]) #direction change
        self._particles[:, 2] %= 2 * np.pi

        # move in the (noisy) commanded direction
        dist = (u[1] * self._dt) + (randn(self._particles_qty) * std[1]) #velocity*tiempo + noise
        self._particles[:, 0] += np.cos(self._particles[:, 2]) * dist
        self._particles[:, 1] += np.sin(self._particles[:, 2]) * dist

        pos = self._particles[:, 0:2]
        self._center = np.average(pos, weights=self._weights, axis=0)

    def _update_box(self, detection: Detection) -> None:
        x = self._box_to_point(detection.box)
        noise = .1

        error = self._center - x
        self._calc_error_distribution(error)

        zs = np.linalg.norm(error, axis=0) + (randn(1) * noise)
        distance = np.linalg.norm(self._particles[:, 0:2] - zs, axis=1)
        self._weights *= scipy.stats.norm(distance, noise).pdf(zs)
        self._weights += 1.e-300 # avoid round-off to zero
        self._weights /= sum(self._weights) # normalize

        # resample if too few effective particles
        if (1. / np.sum(np.square(self._weights))) < self._particles_qty/2:
            indexes = systematic_resample(self._weights)
            self._particles[:] = self._particles[indexes]
            self._particles_qty = len(self._particles)
            self._weights.resize(self._particles_qty)
            self._weights.fill (1.0 / len(self._weights))
            assert np.allclose(self._weights, 1/self._particles_qty)

        self._center = x

    def _create_gaussian_particles(self, mean, std, N):
        particles = np.empty((N, 3))
        particles[:, 0] = mean[0] + (randn(N) * std[0])
        particles[:, 1] = mean[1] + (randn(N) * std[1])
        particles[:, 2] = mean[2] + (randn(N) * std[2])
        particles[:, 2] %= 2 * np.pi
        return particles

class UnscentedKalmanTracker(Tracker):
    """ A single object tracker using Unscented Kalman filter """

    def __init__(self,
                 model_kwargs: dict = { 'dt': 1 },
                 x0: Optional[Vector] = None,
                 box0: Optional[Box] = None,
                 **kwargs) -> None:
        
        super(UnscentedKalmanTracker, self).__init__(box0, **kwargs)

        self.model_kwargs: dict = model_kwargs
        self.model = Model(**self.model_kwargs)

        # Generate 2n+1 points
        x = self.model.box_to_x(box0)
        sigmas = MerweScaledSigmaPoints(n=len(x), alpha=.1, beta=2., kappa=3-model_kwargs['dim_pos'])

        def fx(x, dt):
            # state transition function - predict next state based on constant velocity model x = vt + x_0
            F = self.model.build_F()
            return np.dot(F, x)
        
        def hx(x):
            # takes a state variable and returns the measurement that would correspond to that state
            H = self.model.build_H()
            return np.dot(H, x)

        tracker = UnscentedKalmanFilter(dim_x=self.model.state_length, dim_z=self.model.measurement_length, dt=model_kwargs['dt'], hx=hx, fx=fx, points=sigmas)

        tracker.Q = self.model.build_Q()
        tracker.R = self.model.build_R()
        #tracker.R = None
        tracker.P = self.model.build_P()
        tracker.x = x

        self._tracker: UnscentedKalmanFilter = tracker

    def _predict(self) -> None:
        self._tracker.predict()
        self._center = self._box_to_point(self.model.x_to_box(self._tracker.x))

    def _update_box(self, detection: Detection) -> None:
        error = self._tracker.x - self.model.box_to_x(detection.box)
        self._calc_error_distribution(error)

        z = self.model.box_to_z(detection.box)
        self._tracker.update(z)
        self._center = self._box_to_point(detection.box)
