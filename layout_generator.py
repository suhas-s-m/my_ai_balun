"""
layout_generator.py - Automated PCB Layout Generator for Microwave / mmWave Baluns
Generates:
 1. Standard GDSII (.gds) files using gdstk for Keysight ADS Momentum/RFPro and HFSS.
 2. Merged DRC-clean copper polygons using Shapely (handling pixel overlaps).
 3. High-resolution visual layout rendering (.png).
 4. Supports both Broadband Distributed Balun and Pixelated Inverse-Designed Balun.
"""

import os
import numpy as np
import gdstk
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
from typing import Tuple, List, Optional
from pcb_stackup import PCBTechnology, AVAILABLE_SUBSTRATES, C_0


# Standard GDSII Layer Mapping for Keysight ADS
LAYER_TOP_METAL = (1, 0)       # Layer 1: Top copper (cond)
LAYER_BOTTOM_METAL = (2, 0)    # Layer 2: Bottom ground/slot (cond2)
LAYER_VIAS = (3, 0)            # Layer 3: Ground vias (hole)
LAYER_PORTS = (4, 0)           # Layer 4: Port pins / boundaries (pin)


# =====================================================================
# 1. Pixelated Balun Layout Generator (from Binary Matrix)
# =====================================================================
def generate_pixelated_balun_layout(
    grid: np.ndarray,
    tech: PCBTechnology,
    pixel_size_um: float = 120.0,
    overlap_ratio: float = 0.20,
    feed_length_um: float = 1000.0,
    cell_name: str = "PIXELATED_BALUN",
    output_gds_path: str = "pixelated_balun.gds",
    output_png_path: Optional[str] = "pixelated_balun_layout.png"
) -> Tuple[gdstk.Cell, float, float]:
    """
    Converts a binary grid into a physical PCB layout:
      - Each '1' pixel becomes a metal rectangle of size (pixel_size * (1 + overlap_ratio))
        to ensure solid electrical and diagonal contact (as in Karahan et al. JSSC 2023).
      - Overlapping pixels are merged into contiguous polygons using Shapely.
      - 50-Ohm feed lines are automatically attached at Port 1 (Input), Port 2 (Out +), Port 3 (Out -).
    """
    n_rows, n_cols = grid.shape
    w_50_um = tech.synth_microstrip_width(50.0)[0] * 1e6
    p_step = pixel_size_um
    p_box = pixel_size_um * (1.0 + overlap_ratio)
    offset = (p_box - p_step) / 2.0

    # Create Shapely boxes for each metal pixel
    pixel_boxes = []
    for r in range(n_rows):
        # r = 0 is top row; y increases downwards in matrix, upwards in Cartesian
        y_center = (n_rows - 1 - r) * p_step
        for c in range(n_cols):
            if grid[r, c] == 1:
                x_center = c * p_step
                b = box(x_center - offset, y_center - offset,
                        x_center + p_step + offset, y_center + p_step + offset)
                pixel_boxes.append(b)

    # Merge touching/overlapping pixels into unified polygons
    merged = unary_union(pixel_boxes)

    # Core active bounding dimensions
    core_w = n_cols * p_step
    core_h = n_rows * p_step

    # Feed line definitions
    # Port 1: Left edge center
    p1_y = (n_rows / 2.0) * p_step
    feed1 = box(-feed_length_um, p1_y - w_50_um / 2.0, 0.0, p1_y + w_50_um / 2.0)

    # Port 2: Right edge upper
    p2_y = (3.0 * n_rows / 4.0) * p_step
    feed2 = box(core_w, p2_y - w_50_um / 2.0, core_w + feed_length_um, p2_y + w_50_um / 2.0)

    # Port 3: Right edge lower
    p3_y = (n_rows / 4.0) * p_step
    feed3 = box(core_w, p3_y - w_50_um / 2.0, core_w + feed_length_um, p3_y + w_50_um / 2.0)

    # Merge core polygons with feed lines
    full_layout_geom = unary_union([merged, feed1, feed2, feed3])

    # Build GDSII Cell using gdstk
    lib = gdstk.Library(unit=1e-6, precision=1e-9)  # Coordinates in micrometers
    cell = lib.new_cell(cell_name)

    # Helper to add Shapely geometry to GDS cell
    def add_geom_to_cell(geom, layer_dt):
        if geom.is_empty:
            return
        if isinstance(geom, Polygon):
            coords = list(geom.exterior.coords)[:-1]
            cell.add(gdstk.Polygon(coords, layer=layer_dt[0], datatype=layer_dt[1]))
            # Handle holes if any
            for interior in geom.interiors:
                hole_coords = list(interior.coords)[:-1]
                cell.add(gdstk.Polygon(hole_coords, layer=layer_dt[0], datatype=layer_dt[1]))
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                add_geom_to_cell(poly, layer_dt)

    add_geom_to_cell(full_layout_geom, LAYER_TOP_METAL)

    # Add Port Pin Markers (Layer 4) for Keysight ADS
    pin_len = 100.0  # 100 um
    # Port 1 Pin
    cell.add(gdstk.rectangle((-feed_length_um, p1_y - w_50_um/2.0),
                             (-feed_length_um + pin_len, p1_y + w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))
    # Port 2 Pin
    cell.add(gdstk.rectangle((core_w + feed_length_um - pin_len, p2_y - w_50_um/2.0),
                             (core_w + feed_length_um, p2_y + w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))
    # Port 3 Pin
    cell.add(gdstk.rectangle((core_w + feed_length_um - pin_len, p3_y - w_50_um/2.0),
                             (core_w + feed_length_um, p3_y + w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))

    # Export GDS file
    lib.write_gds(output_gds_path)
    print(f"[GDSII Layout] Saved layout to {output_gds_path}")

    # Generate PNG preview
    if output_png_path:
        render_layout_png(
            full_geom=full_layout_geom,
            title=f"Inverse-Designed Pixelated Balun ({tech.name})",
            output_path=output_png_path,
            ports=[
                (-feed_length_um, p1_y, "Port 1 (Single-Ended Input)"),
                (core_w + feed_length_um, p2_y, "Port 2 (Balanced +)"),
                (core_w + feed_length_um, p3_y, "Port 3 (Balanced -)")
            ]
        )

    total_w_mm = (core_w + 2 * feed_length_um) / 1000.0
    total_h_mm = core_h / 1000.0
    return cell, total_w_mm, total_h_mm


# =====================================================================
# 2. Broadband Distributed Tapered Balun Layout Generator
# =====================================================================
def generate_broadband_balun_layout(
    tech: PCBTechnology,
    f_min_hz: float = 1.0e9,
    f_max_hz: float = 40.0e9,
    feed_length_mm: float = 1.5,
    output_gds_path: str = "broadband_balun.gds",
    output_png_path: Optional[str] = "broadband_balun_layout.png"
) -> Tuple[gdstk.Cell, float, float]:
    """
    Generates an ultra-wideband (1-40 GHz) distributed mode-converting balun layout:
      - 50-Ohm microstrip input feed line.
      - Smooth exponential/Chebyshev impedance & mode-converting taper section.
      - Symmetrical differential branch split into two 50-Ohm balanced lines.
      - Compatible with Rogers RO4350B, RO4003C, RT/duroid 5880.
    """
    w_50_m, eps_0 = tech.synth_microstrip_width(50.0)
    w_50_um = w_50_m * 1e6
    feed_len_um = feed_length_mm * 1e3

    # Balun transition length optimized for multi-octave bandwidth
    # Geometric center frequency: sqrt(1 * 40) = 6.32 GHz
    f_geom = np.sqrt(f_min_hz * f_max_hz)
    lambda_g_geom_um = (C_0 / (f_geom * np.sqrt(eps_0))) * 1e6
    # Active balun core length = lambda_g / 4 at geometric center
    taper_len_um = lambda_g_geom_um / 4.0
    # Minimum spacing between differential outputs
    diff_pitch_um = max(1000.0, 2.5 * w_50_um)

    polygons_to_union = []

    # 1. Port 1 Input Microstrip Feed Line
    p1_feed = box(0.0, -w_50_um / 2.0, feed_len_um, w_50_um / 2.0)
    polygons_to_union.append(p1_feed)

    # 2. Smooth Tapered Transformation Section (x from feed_len_um to feed_len_um + taper_len_um)
    n_steps = 60
    x_taper = np.linspace(feed_len_um, feed_len_um + taper_len_um, n_steps)
    # Taper width expands smoothly to form the balanced junction
    w_junc_um = diff_pitch_um + w_50_um
    taper_profile = w_50_um + (w_junc_um - w_50_um) * (np.sin(np.pi * (x_taper - feed_len_um) / (2.0 * taper_len_um))**1.5)

    # Upper and lower boundary points of the taper
    top_pts = list(zip(x_taper, taper_profile / 2.0))
    bot_pts = list(zip(x_taper[::-1], -taper_profile[::-1] / 2.0))
    taper_poly = Polygon(top_pts + bot_pts)
    polygons_to_union.append(taper_poly)

    # 3. Central Isolation Slot (Creates the balanced differential mode split)
    slot_len_um = taper_len_um * 0.70
    slot_x_start = feed_len_um + taper_len_um - slot_len_um
    slot_x_end = feed_len_um + taper_len_um + 10.0
    slot_w_start = 50.0  # 50 um narrow slot
    slot_w_end = diff_pitch_um - w_50_um  # expands to differential separation

    x_slot = np.linspace(slot_x_start, slot_x_end, 40)
    slot_prof = slot_w_start + (slot_w_end - slot_w_start) * ((x_slot - slot_x_start) / slot_len_um)**2.0
    slot_top = list(zip(x_slot, slot_prof / 2.0))
    slot_bot = list(zip(x_slot[::-1], -slot_prof[::-1] / 2.0))
    slot_poly = Polygon(slot_top + slot_bot)

    # 4. Symmetrical Differential Output Arms (Ports 2 & 3)
    x_out_start = feed_len_um + taper_len_um
    x_out_end = x_out_start + feed_len_um
    y_p2 = diff_pitch_um / 2.0
    y_p3 = -diff_pitch_um / 2.0

    arm2 = box(x_out_start, y_p2 - w_50_um / 2.0, x_out_end, y_p2 + w_50_um / 2.0)
    arm3 = box(x_out_start, y_p3 - w_50_um / 2.0, x_out_end, y_p3 + w_50_um / 2.0)
    polygons_to_union.extend([arm2, arm3])

    # Merge all metal shapes and subtract the central slot
    metal_union = unary_union(polygons_to_union)
    balun_copper = metal_union.difference(slot_poly)

    # Build GDSII
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("BROADBAND_BALUN")

    def add_geom_to_cell(geom, layer_dt):
        if geom.is_empty:
            return
        if isinstance(geom, Polygon):
            coords = list(geom.exterior.coords)[:-1]
            cell.add(gdstk.Polygon(coords, layer=layer_dt[0], datatype=layer_dt[1]))
            for interior in geom.interiors:
                cell.add(gdstk.Polygon(list(interior.coords)[:-1], layer=layer_dt[0], datatype=layer_dt[1]))
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                add_geom_to_cell(poly, layer_dt)

    add_geom_to_cell(balun_copper, LAYER_TOP_METAL)

    # Add Port Pins (Layer 4)
    pin_len = 100.0
    # Port 1
    cell.add(gdstk.rectangle((0.0, -w_50_um/2.0), (pin_len, w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))
    # Port 2
    cell.add(gdstk.rectangle((x_out_end - pin_len, y_p2 - w_50_um/2.0), (x_out_end, y_p2 + w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))
    # Port 3
    cell.add(gdstk.rectangle((x_out_end - pin_len, y_p3 - w_50_um/2.0), (x_out_end, y_p3 + w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))

    lib.write_gds(output_gds_path)
    print(f"[GDSII Layout] Saved broadband balun layout to {output_gds_path}")

    # Generate PNG preview
    if output_png_path:
        render_layout_png(
            full_geom=balun_copper,
            title=f"Ultra-Wideband Distributed Balun (1-40 GHz) on {tech.name}",
            output_path=output_png_path,
            ports=[
                (0.0, 0.0, "Port 1 (50 Ohm Unbalanced In)"),
                (x_out_end, y_p2, "Port 2 (Balanced +)"),
                (x_out_end, y_p3, "Port 3 (Balanced -)")
            ]
        )

    total_w_mm = x_out_end / 1000.0
    total_h_mm = (diff_pitch_um + w_50_um) / 1000.0
    return cell, total_w_mm, total_h_mm


# =====================================================================
# 3. Helper: Render Layout PNG
# =====================================================================
def render_layout_png(
    full_geom,
    title: str,
    output_path: str,
    ports: List[Tuple[float, float, str]]
):
    """Renders a clean high-resolution engineering plot of the PCB layout."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)

    def plot_polygon(p):
        ext = np.array(p.exterior.coords)
        poly_patch = MplPolygon(ext / 1000.0, facecolor='#c88a32', edgecolor='#6e4b16', lw=1.2, alpha=0.9, label='Top Copper')
        ax.add_patch(poly_patch)
        for interior in p.interiors:
            hole = np.array(interior.coords)
            hole_patch = MplPolygon(hole / 1000.0, facecolor='white', edgecolor='#6e4b16', lw=1.0)
            ax.add_patch(hole_patch)

    if isinstance(full_geom, Polygon):
        plot_polygon(full_geom)
    elif isinstance(full_geom, MultiPolygon):
        for poly in full_geom.geoms:
            plot_polygon(poly)

    # Mark Ports
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for idx, (px, py, label) in enumerate(ports):
        c = colors[idx % len(colors)]
        ax.plot(px / 1000.0, py / 1000.0, marker='o', markersize=9, color=c, markeredgecolor='black', zorder=10)
        offset_y = 0.35 if py >= 0 else -0.35
        ax.annotate(label, xy=(px / 1000.0, py / 1000.0), xytext=(px / 1000.0, (py / 1000.0) + offset_y),
                    fontsize=9, fontweight='bold', color=c,
                    ha='center', arrowprops=dict(arrowstyle="->", color=c, lw=1.2))

    ax.autoscale()
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("X Position (mm)", fontsize=11)
    ax.set_ylabel("Y Position (mm)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Layout preview saved to {output_path}")


# =====================================================================
# 4. Hyperlabs-Style Balun Test Board Layout (Edge-Launch 2.92mm / SMA)
# =====================================================================
def generate_hyperlabs_balun_board_layout(
    tech: PCBTechnology,
    conn_pitch_mm: float = 9.525,  # 0.375" standard edge-launch connector clearance
    launch_len_mm: float = 2.5,
    bend_len_mm: float = 6.0,
    output_gds_path: str = "hyperlabs_balun_40ghz.gds",
    output_png_path: Optional[str] = "hyperlabs_balun_layout.png"
) -> Tuple[gdstk.Cell, float, float]:
    """
    Generates a full test-board layout optimized for edge-launch connectors (SMA / 2.92mm K-connectors):
      - Port 1 at left board edge with GCPW launch pads and ground fencing.
      - Core ultra-wideband balun transformation section.
      - Symmetrical, length-matched differential routing arms with S-bends to expand to
        the standard 9.525 mm (0.375") center-to-center connector pitch so two end-launch
        jacks mount side-by-side without mechanical collision.
      - Exactly equal physical path lengths (L2 = L3) to maintain 0 ps skew and 0 deg layout phase error.
    """
    w_50_m, eps_0 = tech.synth_microstrip_width(50.0)
    w_50_um = w_50_m * 1e6
    gcpw_gap_um = 200.0  # 200 um coplanar ground gap

    launch_len_um = launch_len_mm * 1e3
    bend_len_um = bend_len_mm * 1e3
    target_pitch_um = conn_pitch_mm * 1e3

    # Balun core dimensions
    f_geom = np.sqrt(1.0e9 * 40.0e9)
    lambda_geom_um = (C_0 / (f_geom * np.sqrt(eps_0))) * 1e6
    core_len_um = lambda_geom_um / 4.0
    core_pitch_um = max(800.0, 2.5 * w_50_um)

    polygons_top = []
    polygons_bottom = []
    via_list = []

    # 1. Port 1 Launch (x: 0 to launch_len_um)
    p1_trace = box(0.0, -w_50_um / 2.0, launch_len_um, w_50_um / 2.0)
    polygons_top.append(p1_trace)

    # 2. Balun Core Taper (x: launch_len_um to launch_len_um + core_len_um)
    x_core_start = launch_len_um
    x_core_end = x_core_start + core_len_um
    n_pts = 50
    x_taper = np.linspace(x_core_start, x_core_end, n_pts)
    w_junc_um = core_pitch_um + w_50_um
    taper_prof = w_50_um + (w_junc_um - w_50_um) * (np.sin(np.pi * (x_taper - x_core_start) / (2.0 * core_len_um))**1.5)

    top_taper = list(zip(x_taper, taper_prof / 2.0))
    bot_taper = list(zip(x_taper[::-1], -taper_prof[::-1] / 2.0))
    taper_poly = Polygon(top_taper + bot_taper)

    # Core slotline cut
    slot_len = core_len_um * 0.70
    x_s_start = x_core_end - slot_len
    x_s = np.linspace(x_s_start, x_core_end + 10.0, 30)
    slot_w = 50.0 + (core_pitch_um - w_50_um - 50.0) * ((x_s - x_s_start) / slot_len)**2.0
    s_top = list(zip(x_s, slot_w / 2.0))
    s_bot = list(zip(x_s[::-1], -slot_w[::-1] / 2.0))
    slot_poly = Polygon(s_top + s_bot)
    core_metal = taper_poly.difference(slot_poly)
    polygons_top.append(core_metal)

    # 3. Symmetrical Length-Matched S-Bends (x: x_core_end to x_core_end + bend_len_um)
    # Both arms have mathematically identical arc lengths: L2 == L3
    x_b_start = x_core_end
    x_b_end = x_b_start + bend_len_um
    x_bend = np.linspace(x_b_start, x_b_end, 80)
    # Cosine S-curve profile
    tau = (x_bend - x_b_start) / bend_len_um
    y_shift_profile = 0.5 * (1.0 - np.cos(np.pi * tau))

    # Upper branch centerline (Port 2)
    y2_start = core_pitch_um / 2.0
    y2_end = target_pitch_um / 2.0
    y2_center = y2_start + (y2_end - y2_start) * y_shift_profile

    # Lower branch centerline (Port 3, mirror image)
    y3_center = -y2_center

    # Create polygon ribbons for S-bends
    def make_ribbon(xc, yc, w):
        top_pts = list(zip(xc, yc + w / 2.0))
        bot_pts = list(zip(xc[::-1], yc[::-1] - w / 2.0))
        return Polygon(top_pts + bot_pts)

    arm2_bend = make_ribbon(x_bend, y2_center, w_50_um)
    arm3_bend = make_ribbon(x_bend, y3_center, w_50_um)
    polygons_top.extend([arm2_bend, arm3_bend])

    # 4. Port 2 & Port 3 Output Straight Launches (x: x_b_end to x_b_end + launch_len_um)
    x_out_end = x_b_end + launch_len_um
    arm2_launch = box(x_b_end, y2_end - w_50_um / 2.0, x_out_end, y2_end + w_50_um / 2.0)
    arm3_launch = box(x_b_end, -y2_end - w_50_um / 2.0, x_out_end, -y2_end + w_50_um / 2.0)
    polygons_top.extend([arm2_launch, arm3_launch])

    # 5. GCPW Ground Planes & Vias at Board Edges
    board_h_um = target_pitch_um + 4000.0  # Total board height
    gnd_top_y = board_h_um / 2.0
    gnd_bot_y = -board_h_um / 2.0

    # Input Side Ground Planes
    gnd_in_top = box(0.0, w_50_um / 2.0 + gcpw_gap_um, launch_len_um, gnd_top_y)
    gnd_in_bot = box(0.0, gnd_bot_y, launch_len_um, -w_50_um / 2.0 - gcpw_gap_um)
    polygons_top.extend([gnd_in_top, gnd_in_bot])

    # Output Side Ground Planes (flanking Ports 2 & 3)
    gnd_out_top = box(x_b_end, y2_end + w_50_um / 2.0 + gcpw_gap_um, x_out_end, gnd_top_y)
    gnd_out_mid = box(x_b_end, -y2_end + w_50_um / 2.0 + gcpw_gap_um, x_out_end, y2_end - w_50_um / 2.0 - gcpw_gap_um)
    gnd_out_bot = box(x_b_end, gnd_bot_y, x_out_end, -y2_end - w_50_um / 2.0 - gcpw_gap_um)
    polygons_top.extend([gnd_out_top, gnd_out_mid, gnd_out_bot])

    # Add ground vias (diameter 0.3 mm = 300 um)
    via_d_um = 300.0
    via_r_um = via_d_um / 2.0
    via_x_in = np.linspace(300.0, launch_len_um - 300.0, 4)
    for vx in via_x_in:
        via_list.append((vx, w_50_um / 2.0 + gcpw_gap_um + 400.0))
        via_list.append((vx, -w_50_um / 2.0 - gcpw_gap_um - 400.0))

    via_x_out = np.linspace(x_b_end + 300.0, x_out_end - 300.0, 4)
    for vx in via_x_out:
        via_list.append((vx, y2_end + w_50_um / 2.0 + gcpw_gap_um + 400.0))
        via_list.append((vx, y2_end - w_50_um / 2.0 - gcpw_gap_um - 400.0))
        via_list.append((vx, -y2_end + w_50_um / 2.0 + gcpw_gap_um + 400.0))
        via_list.append((vx, -y2_end - w_50_um / 2.0 - gcpw_gap_um - 400.0))

    # Merge Top Metal
    full_top_metal = unary_union(polygons_top)

    # Bottom Metal (Solid Ground Plane across the entire board)
    bottom_gnd = box(0.0, gnd_bot_y, x_out_end, gnd_top_y)

    # Build GDSII Cell
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("HYPERLABS_BALUN_BOARD")

    def add_geom(geom, layer_dt):
        if geom.is_empty:
            return
        if isinstance(geom, Polygon):
            coords = list(geom.exterior.coords)[:-1]
            cell.add(gdstk.Polygon(coords, layer=layer_dt[0], datatype=layer_dt[1]))
            for interior in geom.interiors:
                cell.add(gdstk.Polygon(list(interior.coords)[:-1], layer=layer_dt[0], datatype=layer_dt[1]))
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                add_geom(p, layer_dt)

    add_geom(full_top_metal, LAYER_TOP_METAL)
    add_geom(bottom_gnd, LAYER_BOTTOM_METAL)

    # Add Vias (Layer 3)
    for vx, vy in via_list:
        via_circ = gdstk.ellipse((vx, vy), radius=via_r_um, layer=LAYER_VIAS[0], datatype=LAYER_VIAS[1])
        cell.add(via_circ)

    # Add Port Pins (Layer 4)
    pin_len = 150.0
    cell.add(gdstk.rectangle((0.0, -w_50_um/2.0), (pin_len, w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))
    cell.add(gdstk.rectangle((x_out_end - pin_len, y2_end - w_50_um/2.0), (x_out_end, y2_end + w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))
    cell.add(gdstk.rectangle((x_out_end - pin_len, -y2_end - w_50_um/2.0), (x_out_end, -y2_end + w_50_um/2.0),
                             layer=LAYER_PORTS[0], datatype=LAYER_PORTS[1]))

    lib.write_gds(output_gds_path)
    print(f"[GDSII Layout] Saved Hyperlabs-style balun board layout to {output_gds_path}")

    # Generate PNG preview
    if output_png_path:
        render_layout_png(
            full_geom=full_top_metal,
            title=f"Hyperlabs-Style 1-40 GHz Balun Board (Pitch = {conn_pitch_mm:.2f} mm / 0.375\" for SMA/2.92mm)",
            output_path=output_png_path,
            ports=[
                (0.0, 0.0, "Port 1 (50 Ohm Unbalanced In)"),
                (x_out_end, y2_end, f"Port 2 (+ Out, {conn_pitch_mm/2:.2f} mm)"),
                (x_out_end, -y2_end, f"Port 3 (- Out, -{conn_pitch_mm/2:.2f} mm)")
            ]
        )

    total_w_mm = x_out_end / 1000.0
    total_h_mm = board_h_um / 1000.0
    return cell, total_w_mm, total_h_mm

