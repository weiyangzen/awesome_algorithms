"""Minimal runnable MVP for the Franck-Hertz experiment."""

from __future__ import annotations

import numpy as np


def simulate_single_electron(
    accelerating_voltage: float,
    tube_length: float,
    mean_free_path: float,
    excitation_energy_ev: float,
    retarding_voltage: float,
    inelastic_probability: float,
    rng: np.random.Generator,
) -> bool:
    """Simulate one electron trajectory; return True if collected."""
    if accelerating_voltage <= 0.0:
        return False

    position = 0.0
    energy_ev = 0.0

    while position < tube_length:
        step = float(rng.exponential(mean_free_path))
        remaining = tube_length - position

        if step >= remaining:
            energy_ev += accelerating_voltage * (remaining / tube_length)
            position = tube_length
            break

        energy_ev += accelerating_voltage * (step / tube_length)
        position += step

        if energy_ev >= excitation_energy_ev and rng.random() < inelastic_probability:
            energy_ev -= excitation_energy_ev

    return energy_ev >= retarding_voltage


def collector_current_ratio(
    accelerating_voltage: float,
    num_electrons: int,
    tube_length: float,
    mean_free_path: float,
    excitation_energy_ev: float,
    retarding_voltage: float,
    inelastic_probability: float,
    rng: np.random.Generator,
) -> float:
    """Estimate normalized collector current by Monte Carlo."""
    collected = 0
    for _ in range(num_electrons):
        if simulate_single_electron(
            accelerating_voltage=accelerating_voltage,
            tube_length=tube_length,
            mean_free_path=mean_free_path,
            excitation_energy_ev=excitation_energy_ev,
            retarding_voltage=retarding_voltage,
            inelastic_probability=inelastic_probability,
            rng=rng,
        ):
            collected += 1

    return collected / num_electrons


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple centered moving average with edge padding."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x_pad, kernel, mode="valid")


def local_maxima_indices(y: np.ndarray) -> np.ndarray:
    """Return indices i such that y[i] is a strict local maximum."""
    if y.size < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def main() -> None:
    # Physical parameters (typical mercury Franck-Hertz scale, simplified model)
    excitation_energy_ev = 4.9
    retarding_voltage = 1.5
    tube_length = 1.0
    mean_free_path = 0.18
    inelastic_probability = 0.92

    # Numerical setup
    num_electrons = 5000
    voltages = np.linspace(0.0, 40.0, 161)

    rng = np.random.default_rng(20260407)

    current = np.array(
        [
            collector_current_ratio(
                accelerating_voltage=float(v),
                num_electrons=num_electrons,
                tube_length=tube_length,
                mean_free_path=mean_free_path,
                excitation_energy_ev=excitation_energy_ev,
                retarding_voltage=retarding_voltage,
                inelastic_probability=inelastic_probability,
                rng=rng,
            )
            for v in voltages
        ],
        dtype=float,
    )

    smooth_current = moving_average(current, window=5)
    peaks = local_maxima_indices(smooth_current)
    peaks = peaks[voltages[peaks] > 4.0]

    peak_voltages = voltages[peaks]
    if peak_voltages.size >= 2:
        peak_spacings = np.diff(peak_voltages)
        mean_spacing = float(np.mean(peak_spacings))
    else:
        peak_spacings = np.array([], dtype=float)
        mean_spacing = float("nan")

    print("Franck-Hertz experiment MVP (Monte Carlo, normalized collector current)")
    print(
        "params: "
        f"E_exc={excitation_energy_ev:.2f} eV, V_r={retarding_voltage:.2f} V, "
        f"L={tube_length:.2f}, lambda={mean_free_path:.2f}, "
        f"p_inelastic={inelastic_probability:.2f}, N={num_electrons}"
    )
    print()
    print("Sampled I_c/I0 values (every ~2.5 V):")

    sample_idx = np.arange(0, voltages.size, 10)
    for i in sample_idx:
        print(
            f"V={voltages[i]:5.2f} V, "
            f"I={current[i]:7.4f}, I_smooth={smooth_current[i]:7.4f}"
        )

    print()
    print("Detected peak voltages (V):")
    if peak_voltages.size == 0:
        print("  <none>")
    else:
        print("  " + ", ".join(f"{v:.2f}" for v in peak_voltages[:8]))

    if peak_spacings.size > 0:
        print("Peak spacings (V): " + ", ".join(f"{d:.2f}" for d in peak_spacings[:8]))
    print(f"Mean peak spacing: {mean_spacing:.3f} V")

    # MVP checks: oscillations should exist and spacing should be near excitation energy.
    if peak_voltages.size < 3:
        raise RuntimeError("Too few peaks detected; adjust Monte Carlo or model parameters")

    spacing_error = abs(mean_spacing - excitation_energy_ev)
    print(f"Spacing error |mean_spacing - E_exc|: {spacing_error:.3f} V")

    if spacing_error > 1.4:
        raise RuntimeError(
            "Peak spacing is too far from excitation energy in this run; "
            "model calibration failed"
        )


if __name__ == "__main__":
    main()
