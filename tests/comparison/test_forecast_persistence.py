"""仅副本允许源帧车辆持续被跟踪；历史零填充仍不可见。"""
import numpy as np

from experiments.comparison.ctrl_sim_forecast import PersistentForecastDataset


class Dataset:
    def get_agent_mask(self, states, normalize_dict, fov=None):
        return np.abs(states[..., 0]) < 40


def test_source_vehicle_outside_fov_is_tracked_but_new_vehicle_is_not():
    states = np.zeros((2, 1, 5))
    states[..., 0] = 60
    original = Dataset()
    view = PersistentForecastDataset(original, [True, False])
    assert view.get_agent_mask(states, {})[:, 0].tolist() == [True, False]
    assert not original.get_agent_mask(states, {}).any()


def test_unobserved_history_is_not_made_valid():
    states = np.ones((1, 4, 5)) * 60
    view = PersistentForecastDataset(Dataset(), [True], lambda: np.array([[False, False, False, True]]))
    assert view.get_agent_mask(states, {}).tolist() == [[False, False, False, True]]


def test_nonfinite_tracking_is_not_repaired():
    states = np.zeros((1, 1, 5))
    states[0, 0, 0] = np.nan
    assert not PersistentForecastDataset(Dataset(), [True]).get_agent_mask(states, {}).any()


def test_zero_padded_history_inside_fov_is_still_unobserved():
    states = np.zeros((1, 4, 5))
    view = PersistentForecastDataset(Dataset(), [True], lambda: np.array([[False, False, False, True]]))
    assert view.get_agent_mask(states, {}).tolist() == [[False, False, False, True]]
