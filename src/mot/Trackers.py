import numpy as np

from filterpy.gh import GHFilter
from motpy.core import Box, Detection, Vector, setup_logger
from motpy.model import Model
from motpy.tracker import SingleObjectTracker
from typing import Optional

logger = setup_logger(__name__)

class GHTracker(SingleObjectTracker):
    """ A single object tracker using GH filter with specified motion model specification """

    def __init__(self,
                 model_kwargs: dict = { 'dt': 1 },
                 x0: Optional[Vector] = None,
                 box0: Optional[Box] = None,
                 **kwargs) -> None:

        super(GHTracker, self).__init__(**kwargs)

        self.model_kwargs: dict = model_kwargs
        self.model = Model(**self.model_kwargs)

        if x0 is None:
            x0 = self.model.box_to_x(box0)

        self._tracker: GHFilter = GHFilter(x=x0, g=0.01, h=0.1, dx=0, dt=model_kwargs['dt'])

    def _predict(self) -> None:
        ##self._tracker.predict()
        #self._tracker.dx_prediction = self._tracker.dx
        #self._tracker.x_prediction  = self._tracker.x + (self._tracker.dx*self._tracker.dt)
        pass

    def _update_box(self, detection: Detection) -> None:
        x = self.model.box_to_x(detection.box)
        self._tracker.update(x)

    def box(self) -> Box:
        return self.model.x_to_box(self._tracker.x)

    def is_invalid(self) -> bool:
        try:
            has_nans = any(np.isnan(self._tracker.x))
            return has_nans
        except Exception as e:
            logger.warning(f'invalid tracker - exception: {e}')
            return True
