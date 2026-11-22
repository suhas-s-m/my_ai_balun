"""
pcb_stackup.py - High-Frequency PCB Substrate Stackup & Transmission Line Engine
Supports 1 GHz to 40 GHz operation with Rogers RO4350B, RO4003C, and RT/duroid 5880.
Calculates microstrip synthesis, effective permittivity, dispersion, skin effect,
attenuation, and higher-order mode cutoff limits.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

C_0 = 299792458.0  # Speed of light in vacuum (m/s)
MU_0 = 4.0 * np.pi * 1e-7  # Vacuum permeability (H/m)


@dataclass
class PCBTechnology:
    name: str
    er: float             # Relative permittivity
    tand: float           # Loss tangent
    h_m: float            # Substrate height in meters
    t_m: float            # Copper thickness in meters
    sigma: float = 5.8e7  # Copper conductivity (S/m)
    roughness_m: float = 0.3e-6  # RMS surface roughness (m)

    @property
    def h_mm(self) -> float:
        return self.h_m * 1e3

    @property
    def h_mil(self) -> float:
        return self.h_m / 25.4e-6

    @property
    def t_um(self) -> float:
        return self.t_m * 1e6

    def surface_wave_cutoff_ghz(self) -> float:
        """
        First surface wave (TM0 / TE1) cutoff / onset of severe substrate radiation.
        fc ~= c / (4 * h * sqrt(er - 1))
        """
        if self.er <= 1.0:
            return float('inf')
        return (C_0 / (4.0 * self.h_m * np.sqrt(self.er - 1.0))) / 1e9

    def synth_microstrip_width(self, z0_target: float = 50.0) -> Tuple[float, float]:
        """
        Synthesizes microstrip trace width W (meters) for target Z0 using Hammerstad-Jensen formula.
        Returns: (W_meters, eps_eff_quasi_static)
        """
        er = self.er
        h = self.h_m
        A = (z0_target / 60.0) * np.sqrt((er + 1.0) / 2.0) + ((er - 1.0) / (er + 1.0)) * (0.23 + 0.11 / er)
        B = (377.0 * np.pi) / (2.0 * z0_target * np.sqrt(er))

        # Initial guess from standard formulas
        if A > 1.52:
            w_over_h = 8.0 * np.exp(A) / (np.exp(2.0 * A) - 2.0)
        else:
            w_over_h = (2.0 / np.pi) * (B - 1.0 - np.log(2.0 * B - 1.0) + ((er - 1.0) / (2.0 * er)) * (np.log(B - 1.0) + 0.39 - 0.61 / er))

        # Refine with Newton / secant iteration for precision
        w = w_over_h * h
        for _ in range(10):
            z0_calc, eps_eff = self.calc_microstrip_z0(w)
            err = z0_calc - z0_target
            if abs(err) < 1e-4:
                break
            # Numerical derivative
            dw = w * 1e-4
            z0_plus, _ = self.calc_microstrip_z0(w + dw)
            dz_dw = (z0_plus - z0_calc) / dw
            w = w - err / dz_dw

        _, eps_eff = self.calc_microstrip_z0(w)
        return w, eps_eff

    def calc_microstrip_z0(self, w_m: float) -> Tuple[float, float]:
        """
        Calculates characteristic impedance Z0 and quasi-static eps_eff for microstrip width W.
        Uses Hammerstad-Jensen analytical equations.
        """
        u = w_m / self.h_m
        er = self.er

        # Quasi-static effective permittivity
        a = 1.0 + (1.0 / 49.0) * np.log((u**4 + (u / 52.0)**2) / (u**4 + 0.432)) + (1.0 / 18.7) * np.log(1.0 + (u / 18.1)**3)
        b = 0.564 * ((er - 0.9) / (er + 3.0))**0.053
        eps_eff = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (1.0 + 10.0 / u)**(-a * b)

        # Characteristic impedance
        f_u = 6.0 + (2.0 * np.pi - 6.0) * np.exp(-(30.666 / u)**0.7528)
        eta_0 = 376.730313  # Free space wave impedance
        z0 = (eta_0 / (2.0 * np.pi * np.sqrt(eps_eff))) * np.log(f_u / u + np.sqrt(1.0 + (2.0 / u)**2))
        return z0, eps_eff

    def dispersion_eps_eff(self, w_m: float, f_hz: np.ndarray) -> np.ndarray:
        """
        Calculates high-frequency dispersion of eps_eff up to 40 GHz using Kirschning-Jansen model.
        """
        f_ghz = np.asarray(f_hz) / 1e9
        u = w_m / self.h_m
        er = self.er
        h_mm = self.h_m * 1e3
        _, eps_eff_0 = self.calc_microstrip_z0(w_m)

        # Kirschning-Jansen dispersion parameters
        p1 = 0.27488 + (0.6315 + 0.525 / (1.0 + 0.157 * f_ghz * h_mm)**20) * u - 0.065683 * np.exp(-8.7513 * u)
        p2 = 0.33622 * (1.0 - np.exp(-0.03442 * er))
        p3 = 0.0363 * np.exp(-4.6 * u) * (1.0 - np.exp(-(f_ghz * h_mm / 3.87)**4.97))
        p4 = 1.0 + 2.751 * (1.0 - np.exp(-(er / 15.916)**8))
        p = p1 * p2 * ((0.1844 + p3 * p4) * 10.0 * f_ghz * h_mm)**1.5763

        eps_eff_f = er - (er - eps_eff_0) / (1.0 + p)
        return eps_eff_f

    def calc_losses(self, w_m: float, f_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates attenuation constants:
        - alpha_c: Conductor attenuation (Np/m) with skin effect & surface roughness
        - alpha_d: Dielectric attenuation (Np/m)
        """
        f = np.asarray(f_hz)
        omega = 2.0 * np.pi * f
        eps_eff_f = self.dispersion_eps_eff(w_m, f)

        # Skin depth: delta = sqrt(2 / (omega * mu * sigma))
        delta = np.sqrt(2.0 / (np.maximum(omega, 1.0) * MU_0 * self.sigma))
        r_s = 1.0 / (self.sigma * delta)  # Surface sheet resistance

        # Roughness correction factor (Morgan's rule)
        k_rough = 1.0 + (2.0 / np.pi) * np.arctan(1.4 * (self.roughness_m / np.maximum(delta, 1e-12))**2)
        r_s_eff = r_s * k_rough

        # Conductor loss alpha_c
        z0, _ = self.calc_microstrip_z0(w_m)
        alpha_c = (r_s_eff / (z0 * w_m)) * (1.0 + self.h_m / w_m)

        # Dielectric loss alpha_d
        k0 = omega / C_0
        alpha_d = (k0 * self.er * (eps_eff_f - 1.0) * self.tand) / (2.0 * np.sqrt(eps_eff_f) * (self.er - 1.0))

        return alpha_c, alpha_d


# Pre-defined High-Frequency Substrates
AVAILABLE_SUBSTRATES = {
    "Rogers_RO4350B": PCBTechnology(
        name="Rogers RO4350B (Standard RF Laminate)",
        er=3.66,
        tand=0.0037,
        h_m=0.168e-3,       # 6.6 mil (used in Sengupta et al. Dall-EM paper)
        t_m=35e-6,          # 1 oz copper
        roughness_m=0.3e-6
    ),
    "Rogers_RO4003C": PCBTechnology(
        name="Rogers RO4003C (Low Loss RF Laminate)",
        er=3.55,
        tand=0.0027,
        h_m=0.203e-3,       # 8 mil
        t_m=17.5e-6,        # 0.5 oz copper
        roughness_m=0.25e-6
    ),
    "Rogers_RT_duroid_5880": PCBTechnology(
        name="Rogers RT/duroid 5880 (Ultra-Low Loss PTFE mmWave)",
        er=2.20,
        tand=0.0009,
        h_m=0.127e-3,       # 5 mil
        t_m=17.5e-6,        # 0.5 oz copper
        roughness_m=0.15e-6
    ),
}


def print_substrate_comparison(z0_target: float = 50.0):
    """Prints a comparison table of PCB substrates evaluated at 1 GHz, 20 GHz, and 40 GHz."""
    print("=" * 82)
    print(" HIGH-FREQUENCY PCB SUBSTRATE COMPARISON FOR 1 GHz - 40 GHz BALUN")
    print("=" * 82)
    test_freqs = np.array([1.0e9, 20.0e9, 40.0e9])

    for key, tech in AVAILABLE_SUBSTRATES.items():
        w_50, eps0 = tech.synth_microstrip_width(z0_target)
        fc_sw = tech.surface_wave_cutoff_ghz()
        eps_f = tech.dispersion_eps_eff(w_50, test_freqs)
        ac, ad = tech.calc_losses(w_50, test_freqs)
        total_att_db_cm = (ac + ad) * 8.686 * 0.01  # Np/m to dB/cm

        print(f"\nSubstrate: {tech.name}")
        print(f"  Dielectric: er = {tech.er}, tand = {tech.tand}")
        print(f"  Thickness: h = {tech.h_mil:.2f} mil ({tech.h_mm:.3f} mm), Cu = {tech.t_um:.1f} um")
        print(f"  50-Ohm Trace Width: W = {w_50*1e3:.3f} mm ({w_50/25.4e-6:.2f} mil)")
        print(f"  Surface Wave Radiation Limit (fc): {fc_sw:.1f} GHz {'(Safe for 40 GHz!)' if fc_sw > 40 else '(Caution: surface waves near 40 GHz)'}")
        print(f"  Attenuation @ 1 GHz  : {total_att_db_cm[0]:.3f} dB/cm (eff_er = {eps_f[0]:.2f})")
        print(f"  Attenuation @ 20 GHz : {total_att_db_cm[1]:.3f} dB/cm (eff_er = {eps_f[1]:.2f})")
        print(f"  Attenuation @ 40 GHz : {total_att_db_cm[2]:.3f} dB/cm (eff_er = {eps_f[2]:.2f})")
    print("=" * 82)
