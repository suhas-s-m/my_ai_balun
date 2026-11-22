"""
balun_metrics.py - RF Performance Metrics and Mixed-Mode Analysis for 3-Port Passive Balun
Supports 1 GHz to 40 GHz ultra-wideband evaluation.
Uses scikit-rf (skrf) and NumPy.
"""

import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


class BalunMetrics:
    """
    Computes and analyzes all key RF performance metrics for a 3-port passive balun:
      Port 1: Single-ended unbalanced input (Z01 = 50 Ohm)
      Port 2: Balanced positive output (+)   (Z02 = 50 Ohm)
      Port 3: Balanced negative output (-)   (Z03 = 50 Ohm)
      Differential port between 2 and 3: Zdiff = 100 Ohm
    """

    def __init__(self, frequencies: np.ndarray, s_params: np.ndarray, z0: float = 50.0):
        """
        Parameters
        ----------
        frequencies : np.ndarray, shape (N,)
            Frequency points in Hz.
        s_params : np.ndarray, shape (N, 3, 3)
            Complex 3-port S-parameter matrix at each frequency.
            Indices are 0-based: Port 1 = 0, Port 2 = 1, Port 3 = 2.
        z0 : float
            Port reference impedance in Ohms (default 50.0).
        """
        self.freqs = np.asarray(frequencies)
        self.s = np.asarray(s_params, dtype=complex)
        self.z0 = z0
        self.n_pts = len(self.freqs)

        if self.s.shape != (self.n_pts, 3, 3):
            raise ValueError(f"s_params shape must be ({self.n_pts}, 3, 3), got {self.s.shape}")

        self._compute_all_metrics()

    def _compute_all_metrics(self):
        """Calculate single-ended, differential, mixed-mode metrics, and CMRR."""
        # Single-ended S-parameters
        self.s11 = self.s[:, 0, 0]
        self.s21 = self.s[:, 1, 0]
        self.s31 = self.s[:, 2, 0]
        self.s22 = self.s[:, 1, 1]
        self.s33 = self.s[:, 2, 2]
        self.s23 = self.s[:, 1, 2]
        self.s32 = self.s[:, 2, 1]

        # dB Magnitudes
        self.s11_db = 20.0 * np.log10(np.maximum(np.abs(self.s11), 1e-12))
        self.s21_db = 20.0 * np.log10(np.maximum(np.abs(self.s21), 1e-12))
        self.s31_db = 20.0 * np.log10(np.maximum(np.abs(self.s31), 1e-12))
        self.s22_db = 20.0 * np.log10(np.maximum(np.abs(self.s22), 1e-12))
        self.s33_db = 20.0 * np.log10(np.maximum(np.abs(self.s33), 1e-12))

        # Insertion Loss (IL) at Port 2 and Port 3 in positive dB
        self.il2_db = -self.s21_db
        self.il3_db = -self.s31_db

        # Excess Insertion Loss beyond ideal 3.01 dB power split:
        # Total output power = |S21|^2 + |S31|^2
        p_out = np.abs(self.s21)**2 + np.abs(self.s31)**2
        self.il_excess_db = -10.0 * np.log10(np.maximum(p_out, 1e-12))

        # Amplitude Imbalance: | |S21|_dB - |S31|_dB |
        self.amp_imbalance_db = np.abs(self.s21_db - self.s31_db)

        # Phase Imbalance: difference between relative phase and ideal 180 degrees
        phase2 = np.angle(self.s21, deg=True)
        phase3 = np.angle(self.s31, deg=True)
        phase_diff = phase2 - phase3
        # Wrap phase difference to [-180, 180] relative to 180 deg
        # Ideal phase_diff is +180 or -180 deg.
        # Deviation = |(phase_diff - 180 + 180) % 360 - 180|
        # Or more cleanly: deviation from odd multiple of 180
        dev = (phase_diff - 180.0 + 180.0) % 360.0 - 180.0
        self.phase_imbalance_deg = np.abs(dev)

        # Mixed-Mode S-parameters
        # Port 1 is single-ended (s); Ports 2 and 3 form balanced differential (d) and common (c)
        # Sds21: Single-ended to Differential transmission = (S21 - S31) / sqrt(2)
        # Scs21: Single-ended to Common transmission       = (S21 + S31) / sqrt(2)
        # Sdd22: Differential output return loss           = (S22 - S23 - S32 + S33) / 2
        # Scc22: Common-mode output return loss            = (S22 + S23 + S32 + S33) / 2
        self.sds21 = (self.s21 - self.s31) / np.sqrt(2.0)
        self.scs21 = (self.s21 + self.s31) / np.sqrt(2.0)
        self.sdd22 = (self.s22 - self.s23 - self.s32 + self.s33) / 2.0
        self.scc22 = (self.s22 + self.s23 + self.s32 + self.s33) / 2.0

        self.sds21_db = 20.0 * np.log10(np.maximum(np.abs(self.sds21), 1e-12))
        self.scs21_db = 20.0 * np.log10(np.maximum(np.abs(self.scs21), 1e-12))
        self.sdd22_db = 20.0 * np.log10(np.maximum(np.abs(self.sdd22), 1e-12))
        self.scc22_db = 20.0 * np.log10(np.maximum(np.abs(self.scc22), 1e-12))

        # Common-Mode Rejection Ratio (CMRR):
        # Ratio of differential gain to common-mode gain
        # CMRR = 20 * log10(|Sds21| / |Scs21|)
        self.cmrr_db = self.sds21_db - self.scs21_db

    def get_summary(self) -> Dict[str, float]:
        """Return scalar worst-case and average summary metrics over entire band."""
        return {
            "S11_max_dB": float(np.max(self.s11_db)),
            "S11_mean_dB": float(np.mean(self.s11_db)),
            "IL_excess_max_dB": float(np.max(self.il_excess_db)),
            "IL_excess_mean_dB": float(np.mean(self.il_excess_db)),
            "Amp_Imbalance_max_dB": float(np.max(self.amp_imbalance_db)),
            "Amp_Imbalance_mean_dB": float(np.mean(self.amp_imbalance_db)),
            "Phase_Imbalance_max_deg": float(np.max(self.phase_imbalance_deg)),
            "Phase_Imbalance_mean_deg": float(np.mean(self.phase_imbalance_deg)),
            "CMRR_min_dB": float(np.min(self.cmrr_db)),
            "CMRR_mean_dB": float(np.mean(self.cmrr_db)),
            "Sds21_mean_dB": float(np.mean(self.sds21_db)),
            "Sdd22_max_dB": float(np.max(self.sdd22_db)),
        }

    def print_report(self, title: str = "Balun Performance Report (1 - 40 GHz)"):
        """Print a formatted report of key metrics."""
        summary = self.get_summary()
        print("=" * 65)
        print(f" {title.center(61)} ")
        print("=" * 65)
        print(f" Frequency Range         : {self.freqs[0]/1e9:.2f} GHz to {self.freqs[-1]/1e9:.2f} GHz ({self.n_pts} points)")
        print(f" Input Return Loss (S11) : Worst = {summary['S11_max_dB']:.2f} dB, Mean = {summary['S11_mean_dB']:.2f} dB")
        print(f" Excess Insertion Loss   : Worst = {summary['IL_excess_max_dB']:.2f} dB, Mean = {summary['IL_excess_mean_dB']:.2f} dB")
        print(f" Amplitude Imbalance     : Worst = {summary['Amp_Imbalance_max_dB']:.2f} dB, Mean = {summary['Amp_Imbalance_mean_dB']:.2f} dB")
        print(f" Phase Imbalance         : Worst = {summary['Phase_Imbalance_max_deg']:.2f} deg, Mean = {summary['Phase_Imbalance_mean_deg']:.2f} deg")
        print(f" Diff Gain (Sds21)       : Mean  = {summary['Sds21_mean_dB']:.2f} dB (ideal: 0.0 dB)")
        print(f" Diff Return Loss (Sdd22): Worst = {summary['Sdd22_max_dB']:.2f} dB")
        print(f" CMRR                    : Min   = {summary['CMRR_min_dB']:.2f} dB, Mean = {summary['CMRR_mean_dB']:.2f} dB")
        print("=" * 65)

    def to_skrf_network(self) -> rf.Network:
        """Convert to a scikit-rf Network object."""
        freq = rf.Frequency.from_f(self.freqs, unit='hz')
        return rf.Network(frequency=freq, s=self.s, z0=self.z0)

    def write_touchstone(self, filename: str):
        """Export Touchstone .s3p file compatible with Keysight ADS / Cadence / HFSS."""
        nw = self.to_skrf_network()
        nw.write_touchstone(filename)
        print(f"[Touchstone] Saved 3-port S-parameters to {filename}")

    def plot_performance(self, save_path: Optional[str] = None):
        """Generate a 4-panel comprehensive plot of all balun metrics."""
        f_ghz = self.freqs / 1e9
        fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=150)

        # Panel 1: S-parameter Magnitudes (Return Loss & Transmission)
        ax = axs[0, 0]
        ax.plot(f_ghz, self.s11_db, 'b-', label='S11 (Input Return Loss)', lw=2)
        ax.plot(f_ghz, self.s21_db, 'r-', label='S21 (Port 1 -> Port 2)', lw=1.8)
        ax.plot(f_ghz, self.s31_db, 'g--', label='S31 (Port 1 -> Port 3)', lw=1.8)
        ax.plot(f_ghz, self.sds21_db, 'm-.', label='Sds21 (Diff Gain)', lw=2)
        ax.axhline(-3.01, color='gray', linestyle=':', label='Ideal 3.01 dB Split')
        ax.axhline(-10.0, color='blue', linestyle=':', alpha=0.6, label='-10 dB Spec')
        ax.set_title("Scattering Parameters (S11, S21, S31, Sds21)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_ylim(-35, 2)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9, loc='lower left')

        # Panel 2: Amplitude and Phase Imbalance
        ax = axs[0, 1]
        ax2 = ax.twinx()
        l1 = ax.plot(f_ghz, self.amp_imbalance_db, 'c-', lw=2, label='Amplitude Imbalance (dB)')
        l2 = ax2.plot(f_ghz, self.phase_imbalance_deg, 'darkorange', lw=2, label='Phase Imbalance (deg)')
        ax.set_title("Imbalance Across Band (1 - 40 GHz)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Amplitude Imbalance (dB)", color='c')
        ax2.set_ylabel("Phase Imbalance (degrees)", color='darkorange')
        ax.set_ylim(0, max(2.0, float(np.max(self.amp_imbalance_db)) * 1.2))
        ax2.set_ylim(0, max(15.0, float(np.max(self.phase_imbalance_deg)) * 1.2))
        ax.grid(True, linestyle='--', alpha=0.6)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=9, loc='upper right')

        # Panel 3: CMRR and Common-Mode Suppression
        ax = axs[1, 0]
        ax.plot(f_ghz, self.cmrr_db, 'darkgreen', lw=2.2, label='CMRR (dB)')
        ax.plot(f_ghz, self.scs21_db, 'r:', lw=1.8, label='Scs21 (Common Mode Gain)')
        ax.axhline(20.0, color='darkgreen', linestyle='--', label='20 dB CMRR Target')
        ax.set_title("Common Mode Rejection Ratio (CMRR)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("CMRR / Gain (dB)")
        ax.set_ylim(-40, 50)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9, loc='lower left')

        # Panel 4: Output Matching and Excess Loss
        ax = axs[1, 1]
        ax.plot(f_ghz, self.sdd22_db, 'indigo', lw=2, label='Sdd22 (Diff Return Loss)')
        ax.plot(f_ghz, self.scc22_db, 'teal', lw=1.8, linestyle='--', label='Scc22 (Common Return Loss)')
        ax.plot(f_ghz, self.il_excess_db, 'crimson', lw=2, label='Excess Insertion Loss (dB)')
        ax.axhline(-10.0, color='gray', linestyle=':', label='-10 dB Match Line')
        ax.set_title("Output Matching & Excess Insertion Loss", fontsize=11, fontweight='bold')
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("dB")
        ax.set_ylim(-35, 5)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9, loc='lower left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Plot] Performance plot saved to {save_path}")
        plt.close()
