"""Monte Carlo model of two-dimensional Brownian motion."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def simulate(n_particles: int = 2500, steps: int = 500, dt: float = 0.02,
             diffusion: float = 0.5, seed: int | None = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return times and positions with shape (steps + 1, n_particles, 2).

    Each Cartesian increment is independent N(0, 2*D*dt).
    """
    if n_particles <= 0:
        raise ValueError("n_particles должно быть положительным")
    if steps <= 0:
        raise ValueError("steps должно быть положительным")
    if dt <= 0:
        raise ValueError("dt должно быть положительным")
    if diffusion <= 0:
        raise ValueError("diffusion должно быть положительным")

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * diffusion * dt)
    increments = rng.normal(0.0, sigma, size=(steps, n_particles, 2))
    positions = np.empty((steps + 1, n_particles, 2), dtype=float)
    positions[0] = 0.0
    positions[1:] = np.cumsum(increments, axis=0)
    times = np.arange(steps + 1) * dt
    return times, positions


def mean_squared_displacement(positions: np.ndarray) -> np.ndarray:
    """Ensemble-averaged squared distance from the initial point."""
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions должны иметь форму (steps, particles, 2)")
    return np.mean(np.sum(positions ** 2, axis=2), axis=1)


def theoretical_msd(times: np.ndarray, diffusion: float) -> np.ndarray:
    """Einstein relation for 2D Brownian motion: <r^2> = 4 D t."""
    return 4.0 * diffusion * np.asarray(times)


def save_plots(times: np.ndarray, positions: np.ndarray, diffusion: float,
               output_dir: str | Path) -> None:
    """Save trajectory, spatial density, MSD and radial distribution plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_show = min(40, positions.shape[1])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_show):
        ax.plot(positions[:, i, 0], positions[:, i, 1], lw=0.8, alpha=0.7)
    ax.scatter([0], [0], marker="*", s=60, label="Начало координат")
    ax.set(title="Траектории броуновских частиц", xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=160); plt.close(fig)

    final = positions[-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    hist = ax.hist2d(final[:, 0], final[:, 1], bins=50, cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="Число частиц")
    ax.set(title=f"Распределение частиц, t = {times[-1]:.2f}",
           xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(out / "heatmap.png", dpi=160); plt.close(fig)

    msd = mean_squared_displacement(positions)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, msd, label="Монте-Карло", lw=2)
    ax.plot(times, theoretical_msd(times, diffusion), "--",
            label="Теория: 4Dt", lw=2)
    ax.set(title="Среднеквадратичное смещение", xlabel="Время t",
           ylabel="MSD = <r²>")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "msd.png", dpi=160); plt.close(fig)

    radius = np.linalg.norm(final, axis=1)
    t = times[-1]
    bins = np.linspace(0, max(radius.max() * 1.05, 1e-9), 45)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(radius, bins=bins, density=True, alpha=0.55, label="Монте-Карло")
    r = np.linspace(0, bins[-1], 400)
    pdf = (r / (2 * diffusion * t)) * np.exp(-r**2 / (4 * diffusion * t))
    ax.plot(r, pdf, label="Теория", lw=2)
    ax.set(title="Радиальное распределение частиц", xlabel="Радиус r",
           ylabel="Плотность вероятности")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "radial_distribution.png", dpi=160); plt.close(fig)



def save_animation(times: np.ndarray, positions: np.ndarray,
                   output_dir: str | Path, fps: int = 12,
                   max_particles: int = 40, max_frames: int = 80) -> Path:
    """Save a GIF animation using the exact positions array from simulate().

    The animation does not run a second simulation: every frame is a slice
    of the same `positions` array used by save_plots() and MSD calculations.
    For readability, at most `max_particles` trajectories are displayed.
    """
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions должны иметь форму (steps, particles, 2)")
    if times.ndim != 1 or len(times) != positions.shape[0]:
        raise ValueError("times должен соответствовать первому измерению positions")
    if len(times) < 2:
        raise ValueError("Для GIF требуется не менее двух временных слоёв")
    if fps <= 0 or max_particles <= 0 or max_frames < 2:
        raise ValueError("fps и max_particles должны быть положительными, max_frames >= 2")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gif_path = out / "brownian_motion.gif"

    n_show = min(max_particles, positions.shape[1])
    # Use the same first particles as the static trajectory plot.
    tracks = positions[:, :n_show, :]
    frame_indices = np.unique(
        np.linspace(0, len(times) - 1, min(max_frames, len(times)), dtype=int)
    )

    x_min, x_max = float(tracks[:, :, 0].min()), float(tracks[:, :, 0].max())
    y_min, y_max = float(tracks[:, :, 1].min()), float(tracks[:, :, 1].max())
    x_pad = max((x_max - x_min) * 0.08, 1e-6)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 6))
    lines = [ax.plot([], [], lw=0.9, alpha=0.65)[0] for _ in range(n_show)]
    particles = ax.scatter([], [], s=18, zorder=3)
    ax.scatter([0], [0], marker="*", s=90, color="red",
               edgecolors="black", linewidths=0.4, label="Начало координат")
    time_label = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         va="top", ha="left",
                         bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Броуновское движение: траектории из расчёта Монте-Карло")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    def update(frame_number: int):
        frame = int(frame_indices[frame_number])
        for particle_index, line in enumerate(lines):
            line.set_data(tracks[:frame + 1, particle_index, 0],
                          tracks[:frame + 1, particle_index, 1])
        particles.set_offsets(tracks[frame])
        time_label.set_text(f"t = {times[frame]:.3f}")
        return [*lines, particles, time_label]

    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 / fps,
        blit=False, repeat=True
    )
    try:
        animation.save(gif_path, writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)

    return gif_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Имитационная модель двумерного броуновского движения (Монте-Карло).")
    parser.add_argument("--particles", type=int, default=2500, help="число частиц")
    parser.add_argument("--steps", type=int, default=500, help="число шагов")
    parser.add_argument("--dt", type=float, default=0.02, help="временной шаг")
    parser.add_argument("--diffusion", type=float, default=0.5, help="коэффициент диффузии")
    parser.add_argument("--seed", type=int, default=42, help="зерно генератора случайных чисел")
    parser.add_argument("--output", type=Path, default=Path("results"), help="папка результатов")
    args = parser.parse_args()

    times, positions = simulate(args.particles, args.steps, args.dt,
                                args.diffusion, args.seed)
    save_plots(times, positions, args.diffusion, args.output)
    gif_path = save_animation(times, positions, args.output)
    msd = mean_squared_displacement(positions)
    expected = theoretical_msd(times, args.diffusion)
    rel_error = abs(msd[-1] - expected[-1]) / expected[-1] * 100
    print(f"Частиц: {args.particles}")
    print(f"Шагов: {args.steps}; конечное время: {times[-1]:.4f}")
    print(f"MSD (модель): {msd[-1]:.6f}")
    print(f"MSD (теория): {expected[-1]:.6f}")
    print(f"Относительное отклонение MSD: {rel_error:.2f}%")
    print(f"Графики и GIF сохранены в: {args.output.resolve()}")
    print(f"GIF-анимация: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
