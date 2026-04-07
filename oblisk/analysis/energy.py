import numpy as np

e_charge = 1.602176634e-19  # C
m_u = 1.66053906660e-27  # kg (atomic mass unit)


def energies_keV_from_xp_m(
    xp_m: np.ndarray,
    mass_number: int,
    charge_state: int,
    *,
    b_field_t: float,
    magnetic_length_m: float,
    drift_length_m: float,
) -> np.ndarray:
    """
    Convert |X'| in meters to kinetic energy in keV for an ion species.
    """
    xp_m_arr = np.asarray(xp_m, dtype=float)
    mass_kg = float(mass_number) * m_u
    charge_c = float(charge_state) * e_charge

    velocity_scale = charge_c * float(b_field_t) * float(magnetic_length_m)
    velocity_scale *= float(magnetic_length_m) / 2.0 + float(drift_length_m)
    velocity_scale /= mass_kg

    velocities = velocity_scale / np.maximum(xp_m_arr, 1e-30)
    energy_j = 0.5 * mass_kg * velocities**2
    return (energy_j / e_charge) / 1e3


def energies_keV_from_xp_px(
    xp_px: np.ndarray,
    mass_number: int,
    charge_state: int,
    *,
    meters_per_pixel: float,
    b_field_t: float,
    magnetic_length_m: float,
    drift_length_m: float,
) -> np.ndarray:
    xp_px_arr = np.asarray(xp_px, dtype=float)
    xp_m = np.abs(xp_px_arr) * float(meters_per_pixel)
    return energies_keV_from_xp_m(
        xp_m=xp_m,
        mass_number=mass_number,
        charge_state=charge_state,
        b_field_t=b_field_t,
        magnetic_length_m=magnetic_length_m,
        drift_length_m=drift_length_m,
    )
