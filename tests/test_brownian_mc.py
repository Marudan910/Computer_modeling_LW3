import numpy as np
import pytest
from src.brownian_mc import simulate, mean_squared_displacement, theoretical_msd


def test_simulation_shape_and_initial_state():
    times, positions = simulate(n_particles=20, steps=15, seed=1)
    assert positions.shape == (16, 20, 2)
    assert times.shape == (16,)
    assert np.allclose(positions[0], 0.0)


def test_reproducibility():
    _, a = simulate(n_particles=10, steps=8, seed=123)
    _, b = simulate(n_particles=10, steps=8, seed=123)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("kwargs", [
    {"n_particles": 0}, {"steps": 0}, {"dt": 0}, {"diffusion": -1}
])
def test_invalid_parameters(kwargs):
    params = {"n_particles": 10, "steps": 10, "dt": 0.1, "diffusion": 0.5}
    params.update(kwargs)
    with pytest.raises(ValueError):
        simulate(**params)


def test_msd_theory():
    t = np.array([0.0, 1.0, 2.0])
    assert np.allclose(theoretical_msd(t, 0.5), [0.0, 2.0, 4.0])


def test_msd_is_nonnegative_and_starts_at_zero():
    _, positions = simulate(n_particles=100, steps=20, seed=2)
    msd = mean_squared_displacement(positions)
    assert msd[0] == 0.0
    assert np.all(msd >= 0.0)


def test_save_animation_from_positions(tmp_path):
    from src.brownian_mc import save_animation

    times, positions = simulate(n_particles=5, steps=5, dt=0.1, seed=4)
    gif_path = save_animation(times, positions, tmp_path, max_particles=5, max_frames=6)
    assert gif_path.exists()
    assert gif_path.name == "brownian_motion.gif"
    assert gif_path.stat().st_size > 0
