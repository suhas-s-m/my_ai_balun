"""
run_balun_framework.py - Master Pipeline for Wideband (1 - 40 GHz) Passive Balun Design
Coordinates:
 1. Substrate analysis & technology selection (Rogers RO4350B, RO4003C, RT/duroid 5880).
 2. Ultra-wideband physical synthesis & full 3-port S-parameter calculation.
 3. Evaluation of Insertion Loss, Amplitude/Phase Imbalance, Sds21, Sdd22, and CMRR.
 4. Touchstone (.s3p) export via scikit-rf.
 5. GDSII Layout synthesis using gdstk & Shapely polygon merging.
 6. Inverse-designed pixelated EM structure synthesis (GA + deep-learning framework from papers).
 7. Keysight ADS Momentum/RFPro integration bundle generation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure current directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcb_stackup import AVAILABLE_SUBSTRATES, print_substrate_comparison
from balun_metrics import BalunMetrics
from balun_synthesis import BroadbandDistributedBalun, PixelatedBalunOptimizer
from layout_generator import (
    generate_broadband_balun_layout,
    generate_pixelated_balun_layout,
    generate_hyperlabs_balun_board_layout
)
from ads_export import generate_ads_layermap, generate_ads_substrate, generate_ads_ael_script


def run_pipeline():
    print("=" * 80)
    print("  PASSIVE BALUN SYNTHESIS & INVERSE DESIGN FRAMEWORK (1 GHz - 40 GHz)")
    print("  Based on Deep Learning & Generative AI EM Synthesis Methodologies")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Substrate Comparison & Technology Selection
    # -------------------------------------------------------------
    print("\n>>> STEP 1: Evaluating PCB Substrate Technologies...")
    print_substrate_comparison(z0_target=50.0)

    # Selected primary substrate: Rogers RO4350B (as used in Princeton Dall-EM paper)
    tech = AVAILABLE_SUBSTRATES["Rogers_RO4350B"]
    print(f"\n[Selected Substrate]: {tech.name}")
    print(f"  Dielectric Permittivity (er): {tech.er}")
    print(f"  Loss Tangent (tand)         : {tech.tand}")
    print(f"  Substrate Height (h)        : {tech.h_mm:.4f} mm ({tech.h_mil:.2f} mil)")
    print(f"  Copper Cladding (t)         : {tech.t_um:.1f} um (1 oz rolled copper)")

    # -------------------------------------------------------------
    # Step 2: Ultra-Wideband (1-40 GHz) Balun Physical Synthesis
    # -------------------------------------------------------------
    print("\n>>> STEP 2: Synthesizing Broadband Balun (1.0 to 40.0 GHz)...")
    freqs = np.linspace(1.0e9, 40.0e9, 101)  # 101 frequency points across 1-40 GHz
    balun_model = BroadbandDistributedBalun(tech=tech, z0_in=50.0, z0_out_diff=100.0)
    s_params = balun_model.evaluate_s_params(freqs)

    # -------------------------------------------------------------
    # Step 3: Key Performance Metrics & Mixed-Mode Analysis
    # -------------------------------------------------------------
    print("\n>>> STEP 3: Computing Key RF Metrics, Mixed-Mode S-parameters & CMRR...")
    metrics = BalunMetrics(frequencies=freqs, s_params=s_params, z0=50.0)
    metrics.print_report(title="Ultra-Wideband 1-40 GHz Balun Performance")

    # Plot metrics
    perf_plot_path = "balun_performance_1_40ghz.png"
    metrics.plot_performance(save_path=perf_plot_path)

    # Export Touchstone .s3p file
    s3p_path = "balun_1_40ghz.s3p"
    metrics.write_touchstone(s3p_path)

    # -------------------------------------------------------------
    # Step 4: GDSII PCB Layout Generation
    # -------------------------------------------------------------
    print("\n>>> STEP 4: Generating Broadband PCB Layout (GDSII & Visual Preview)...")
    gds_broadband_path = "broadband_balun_1_40ghz.gds"
    png_broadband_path = "broadband_balun_layout.png"
    cell_bb, total_w, total_h = generate_broadband_balun_layout(
        tech=tech,
        f_min_hz=1.0e9,
        f_max_hz=40.0e9,
        feed_length_mm=1.5,
        output_gds_path=gds_broadband_path,
        output_png_path=png_broadband_path
    )
    print(f"[Broadband Layout Dimensions]: {total_w:.2f} mm x {total_h:.2f} mm")

    # Generate Hyperlabs-Style Test Board (with SMA / 2.92mm connector pitch & GCPW launch)
    print("\n>>> STEP 4b: Generating Hyperlabs-Style Test Board Layout (Edge-Launch SMA / 2.92mm)...")
    gds_hyperlabs_path = "hyperlabs_balun_40ghz.gds"
    png_hyperlabs_path = "hyperlabs_balun_layout.png"
    cell_hl, hl_w, hl_h = generate_hyperlabs_balun_board_layout(
        tech=tech,
        conn_pitch_mm=9.525,  # 0.375" standard edge-launch connector clearance
        launch_len_mm=2.5,
        bend_len_mm=6.0,
        output_gds_path=gds_hyperlabs_path,
        output_png_path=png_hyperlabs_path
    )
    print(f"[Hyperlabs Board Dimensions]: {hl_w:.2f} mm x {hl_h:.2f} mm (Pitch = 9.525 mm / 0.375\")")

    # -------------------------------------------------------------
    # Step 5: Pixelated Inverse-Design Balun (Paper Methodology)
    # -------------------------------------------------------------
    print("\n>>> STEP 5: Running Pixelated Inverse Design GA Optimization (Papers Flow)...")
    print("  Discretizing into 18x18 grid with multi-objective fitness & surrogate...")
    opt = PixelatedBalunOptimizer(
        grid_size=18,
        freq_points=21,
        f_min=1.0e9,
        f_max=40.0e9,
        pop_size=48,
        generations=30,
        tournament_size=6,
        mutation_rate=0.08,
        elite_count=4,
        enforce_symmetry=True
    )
    best_grid, best_cost = opt.optimize(verbose=True)

    gds_pixel_path = "pixelated_balun.gds"
    png_pixel_path = "pixelated_balun_layout.png"
    cell_pix, pix_w, pix_h = generate_pixelated_balun_layout(
        grid=best_grid,
        tech=tech,
        pixel_size_um=100.0,
        overlap_ratio=0.20,
        conn_pitch_mm=9.525,
        launch_len_mm=2.5,
        bend_len_mm=5.0,
        cell_name="PIXELATED_BALUN",
        output_gds_path=gds_pixel_path,
        output_png_path=png_pixel_path
    )
    print(f"[Pixelated Balun with GND & SMA Dimensions]: {pix_w:.2f} mm x {pix_h:.2f} mm (Pitch = 9.525 mm / 0.375\")")

    # -------------------------------------------------------------
    # Step 6: Keysight ADS Integration Bundle Generation
    # -------------------------------------------------------------
    print("\n>>> STEP 6: Generating Keysight ADS Integration Bundle...")
    generate_ads_layermap(filepath="ads_balun.layermap")
    generate_ads_substrate(tech=tech, filepath="ads_rogers_ro4350b.subst")
    generate_ads_ael_script(
        gds_filename=gds_broadband_path,
        cell_name="BROADBAND_BALUN",
        tech=tech,
        s3p_filename=s3p_path,
        filepath="import_and_simulate_balun.ael"
    )

    print("\n" + "=" * 80)
    print("  FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Generated Artifacts in Workspace:")
    print(f"  1. RF Performance Plot    : {os.path.abspath(perf_plot_path)}")
    print(f"  2. Touchstone S-Parameters: {os.path.abspath(s3p_path)}")
    print(f"  3. Broadband Balun GDSII  : {os.path.abspath(gds_broadband_path)}")
    print(f"  4. Broadband Balun Preview: {os.path.abspath(png_broadband_path)}")
    print(f"  5. Hyperlabs Balun GDSII  : {os.path.abspath(gds_hyperlabs_path)}")
    print(f"  6. Hyperlabs Balun Preview: {os.path.abspath(png_hyperlabs_path)}")
    print(f"  7. Pixelated Balun GDSII  : {os.path.abspath(gds_pixel_path)}")
    print(f"  8. Pixelated Balun Preview: {os.path.abspath(png_pixel_path)}")
    print(f"  9. ADS GDS Layer Map      : {os.path.abspath('ads_balun.layermap')}")
    print(f"  10. ADS Substrate (.subst): {os.path.abspath('ads_rogers_ro4350b.subst')}")
    print(f"  11. ADS Automated Script  : {os.path.abspath('import_and_simulate_balun.ael')}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
