# Ultra-Wideband Passive Balun Framework (1 GHz – 40 GHz)

An open-source Python microwave engineering and inverse-design framework for designing passive baluns converting single-ended signals to balanced differential signals across **1 GHz to 40 GHz**.

Based on deep learning and generative EM synthesis methodologies from:
* *Y. Guo, E. A. Karahan, K. Sengupta et al.*, **"Dall-EM: Generative AI with Diffusion Models for New Design Space Discovery and Target-to-Electromagnetic Structure Synthesis,"** IEEE IMS, 2025.
* *E. A. Karahan, Z. Liu, and K. Sengupta*, **"Deep-Learning-Based Inverse-Designed Millimeter-Wave Passives and Power Amplifiers,"** IEEE Journal of Solid-State Circuits (JSSC), vol. 58, no. 11, Nov. 2023.

---

## 1. Quick Start

Activate the virtual environment and run the pipeline:
```bash
source .venv/bin/activate
python run_balun_framework.py
```

---

## 2. Framework Architecture

| Script / Module | Description |
| :--- | :--- |
| `balun_metrics.py` | 3-Port S-parameters, Insertion Loss ($IL_2, IL_3, IL_{\text{excess}}$), Amplitude & Phase Imbalance, Mixed-mode matrix ($S_{ds21}, S_{cs21}, S_{dd22}, S_{cc22}$), CMRR, and Touchstone `.s3p` export via `scikit-rf`. |
| `pcb_stackup.py` | Substrate definition engine (Rogers RO4350B, RO4003C, RT/duroid 5880). Calculates $50\,\Omega$ trace widths, Kirschning-Jansen dispersion, skin-effect attenuation with Morgan roughness correction, and surface-wave radiation cutoff. |
| `balun_synthesis.py` | Dual synthesis engine: (1) Broadband distributed mode-converting balun model (1-40 GHz), and (2) Pixelated inverse-design Genetic Algorithm (GA) optimizer with PyTorch Deep-CNN forward surrogate (`BalunSurrogateCNN`). |
| `layout_generator.py` | GDSII layout synthesis using `gdstk` and `shapely`. Produces DRC-clean merged copper polygons, length-matched differential routing arms, and test-board layouts with connector pitch. |
| `ads_export.py` | Keysight ADS integration bundle: GDS layer map (`.layermap`), substrate definition (`.subst`), and automated AEL script (`.ael`). |
| `run_balun_framework.py` | Master pipeline executing the entire synthesis, evaluation, layout generation, and ADS export workflow. |

---

## 3. Connector & Hardware Test Board (Hyperlabs Style)

For test and measurement up to 40 GHz:
* **Connectors:** 2.92 mm (K-type) edge-launch connectors (100% mechanically intermateable with standard SMA cables, with mode-free operation up to 40 GHz).
* **Connector Clearance Pitch:** $9.525\text{ mm}$ ($0.375\text{"}$) center-to-center spacing on differential Ports 2 & 3 to allow two end-launch connector bodies to mount side-by-side without mechanical collision.
* **Length Matching:** S-bend differential routing arms have mathematically identical physical path lengths ($L_2 \equiv L_3$) to ensure $0\text{ ps}$ delay skew and $0^\circ$ layout phase error.
* **Launch Interface:** Grounded Coplanar Waveguide (GCPW) launches with ground via fences.

---

## 4. Keysight ADS Integration Steps

1. **Import GDSII Layout into ADS Momentum/RFPro:**
   - In ADS: `File -> Import -> Design...`
   - File Type: `GDSII / Stream`
   - Select `hyperlabs_balun_40ghz.gds` (or `broadband_balun_1_40ghz.gds`).
   - In Layer Map, select `ads_balun.layermap`.
2. **Assign Substrate Stackup:**
   - In ADS Momentum / Substrate Editor: `File -> Import...`
   - Select `ads_rogers_ro4350b.subst`.
3. **Simulate Pre-Calculated Benchmark in ADS Schematic:**
   - Place an `S3P` component (`Palette -> Data Items -> S3P`).
   - Set file path to `balun_1_40ghz.s3p`.
   - Terminate Port 1 with $50\,\Omega$, Port 2 with $50\,\Omega$, Port 3 with $50\,\Omega$ (or a $100\,\Omega$ differential load across Ports 2 & 3).
