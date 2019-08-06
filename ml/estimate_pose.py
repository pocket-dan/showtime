from typing import List

import numpy as np

from tf_pose.estimator import Human, TfPoseEstimator
from tf_pose.networks import get_graph_path

WIDTH: int = 432
HEIGHT: int = 368

# ["cmu", "mobilenet_thin", "mobilenet_v2_large", "mobilenet_v2_small"],

MODEL: str = "mobilenet_thin"
RESIZE_OUT_RATIO: float = 4.0

estimator = TfPoseEstimator(get_graph_path(MODEL), target_size=(WIDTH, HEIGHT))


def main(image: np.ndarray) -> List[Human]:
    humans = estimator.inference(
        image,
        resize_to_default=(WIDTH > 0 and HEIGHT > 0),
        upsample_size=RESIZE_OUT_RATIO,
    )

    return humans
