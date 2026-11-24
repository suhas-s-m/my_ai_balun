"""
generate_stackup_and_layout_figures.py - Generates visual cross-section and true 2-layer layout figures.
Shows:
 1. 2-Layer PCB Stackup Cross-Section (Top Signal, Dielectric, Bottom Solid Ground).
 2. Clean 2-Layer Balun Layout with Layer 1 (Top Microstrip & Pixels) and Layer 2 (Bottom Solid GND).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon, FancyArrowPatch, ConnectionPatch
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

from pcb_stackup import AVAILABLE_SUBSTRATES
from layout_generator import LAYER_TOP_METAL, LAYER_BOTTOM_METAL, LAYER_VIAS, LAYER_PORTS


def generate_stackup_diagram(output_path: str = "pcb_stackup_2layer.png"):
    """Generates an annotated engineering cross-section diagram of the 2-layer PCB stackup."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)

    # Substrate parameters (Rogers RO4350B)
    # Heights in relative units for visualization
    h_cu = 0.35      # 35 um Top/Bottom Copper
    h_sub = 1.68     # 168 um (6.6 mil) Rogers RO4350B dielectric
    x_len = 10.0

    y_bottom_gnd = 0.0
    y_sub_bottom = y_bottom_gnd + h_cu
    y_sub_top = y_sub_bottom + h_sub
    y_top_cu = y_sub_top

    # 1. Layer 2: Bottom Solid Ground Plane (Continuous Copper)
    rect_bot_cu = Rectangle(
        (0.5, y_bottom_gnd), x_len, h_cu,
        facecolor='#b45309', edgecolor='#78350f', lw=1.5,
        label='Layer 2: Bottom Copper (Solid GND Plane, 35 µm / 1 oz)'
    )
    ax.add_patch(rect_bot_cu)

    # 2. Dielectric: Rogers RO4350B Core
    rect_sub = Rectangle(
        (0.5, y_sub_bottom), x_len, h_sub,
        facecolor='#e2e8f0', edgecolor='#64748b', lw=1.2,
        label='Dielectric Core: Rogers RO4350B (h = 6.6 mil / 168 µm, εr = 3.66, tanδ = 0.0037)'
    )
    ax.add_patch(rect_sub)

    # 3. Layer 1: Top Copper (Microstrip Traces & Pixelated Balun)
    # Trace 1: 50-Ohm Input Feed
    rect_t1 = Rectangle((1.5, y_top_cu), 1.8, h_cu, facecolor='#d97706', edgecolor='#78350f', lw=1.5)
    ax.add_patch(rect_t1)
    ax.text(2.4, y_top_cu + h_cu + 0.12, "50 Ω Feed\n(W = 0.368 mm)",
            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#92400e')

    # Trace 2: Pixelated Balun Discretized Matrix
    rect_pix1 = Rectangle((4.0, y_top_cu), 0.6, h_cu, facecolor='#d97706', edgecolor='#78350f', lw=1.2)
    rect_pix2 = Rectangle((4.7, y_top_cu), 0.6, h_cu, facecolor='#d97706', edgecolor='#78350f', lw=1.2)
    rect_pix3 = Rectangle((5.4, y_top_cu), 0.6, h_cu, facecolor='#d97706', edgecolor='#78350f', lw=1.2)
    rect_pix4 = Rectangle((6.1, y_top_cu), 0.6, h_cu, facecolor='#d97706', edgecolor='#78350f', lw=1.2)
    for p in [rect_pix1, rect_pix2, rect_pix3, rect_pix4]:
        ax.add_patch(p)
    ax.text(5.35, y_top_cu + h_cu + 0.12, "Pixelated Matrix\n(18x18, 100 µm Pixels)",
            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#92400e')

    # Trace 3: Differential Balanced Outputs
    rect_t2 = Rectangle((7.4, y_top_cu), 1.0, h_cu, facecolor='#d97706', edgecolor='#78350f', lw=1.5)
    rect_t3 = Rectangle((8.7, y_top_cu), 1.0, h_cu, facecolor='#d97706', edgecolor='#78350f', lw=1.5)
    ax.add_patch(rect_t2)
    ax.add_patch(rect_t3)
    ax.text(7.9, y_top_cu + h_cu + 0.12, "Port 2 (+)\n(50 Ω)", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#92400e')
    ax.text(9.2, y_top_cu + h_cu + 0.12, "Port 3 (-)\n(50 Ω)", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#92400e')

    # Dimension and Layer Annotations
    # Layer 1 Tag
    ax.annotate(
        "LAYER 1 (TOP COPPER, 35 µm):\n• 50 Ω Microstrip Feeds (W = 0.368 mm)\n• 18x18 Pixelated Balun Matrix\n• SMA Center Launch Pads",
        xy=(0.5, y_top_cu + h_cu / 2.0), xytext=(-3.8, y_top_cu + 0.2),
        fontsize=8.0, fontweight='bold', color='#92400e',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='#fef3c7', edgecolor='#d97706', lw=1.0),
        arrowprops=dict(arrowstyle="->", color='#d97706', lw=1.5)
    )

    # Dielectric Tag
    ax.annotate(
        "DIELECTRIC SUBSTRATE:\n• Rogers RO4350B\n• Height: h = 6.61 mil (0.168 mm)\n• εr = 3.66, tanδ = 0.0037\n• TM0 Cutoff: 273.5 GHz (Safe)",
        xy=(0.5, y_sub_bottom + h_sub / 2.0), xytext=(-3.8, y_sub_bottom + 0.1),
        fontsize=8.0, fontweight='bold', color='#1e293b',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='#f1f5f9', edgecolor='#64748b', lw=1.0),
        arrowprops=dict(arrowstyle="->", color='#64748b', lw=1.5)
    )

    # Layer 2 Tag
    ax.annotate(
        "LAYER 2 (BOTTOM COPPER, 35 µm):\n• Continuous Solid Ground Plane (GND)\n• Full low-impedance RF return path\n• Bottom solder contact for SMA jacks",
        xy=(0.5, y_bottom_gnd + h_cu / 2.0), xytext=(-3.8, y_bottom_gnd - 0.7),
        fontsize=8.0, fontweight='bold', color='#78350f',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='#fef3c7', edgecolor='#b45309', lw=1.0),
        arrowprops=dict(arrowstyle="->", color='#b45309', lw=1.5)
    )

    # SMA Connector Clamping Annotation (Right Edge)
    rect_sma_pin = Rectangle((9.8, y_top_cu + h_cu), 0.7, 0.25, facecolor='#2563eb', edgecolor='#1d4ed8')
    rect_sma_body_top = Rectangle((10.1, y_top_cu + h_cu + 0.35), 0.9, 0.4, facecolor='#64748b', edgecolor='#334155')
    rect_sma_body_bot = Rectangle((10.1, y_bottom_gnd - 0.45), 0.9, 0.4, facecolor='#64748b', edgecolor='#334155')
    ax.add_patch(rect_sma_pin)
    ax.add_patch(rect_sma_body_top)
    ax.add_patch(rect_sma_body_bot)
    ax.plot([10.55, 10.55], [y_bottom_gnd - 0.05, y_top_cu + h_cu + 0.35], color='#475569', lw=2.5)
    ax.text(10.55, y_top_cu + 1.25, "Edge-Launch SMA / 2.92mm\nJack Clamping Top & Bottom",
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e3a8a')

    # Formatting
    ax.set_xlim(-4.2, 12.0)
    ax.set_ylim(-1.3, 3.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("2-Layer PCB Stackup Cross-Section (Rogers RO4350B, 6.6 mil)", fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Stackup Figure] Saved 2-layer cross-section to {output_path}")


def generate_clean_2layer_layout(
    grid: np.ndarray,
    output_path: str = "pixelated_balun_2layer_layout.png"
):
    """
    Generates a dedicated two-panel layout figure:
      Panel A: Layer 1 (Top Metal) - Microstrip Signal Traces & Pixelated Balun Core ONLY.
      Panel B: Layer 2 (Bottom Metal) - Continuous Solid Copper Ground Plane.
    """
    tech = AVAILABLE_SUBSTRATES["Rogers_RO4350B"]
    w_50_m, _ = tech.synth_microstrip_width(50.0)
    w_50_um = w_50_m * 1e6

    n_rows, n_cols = grid.shape
    p_step = 100.0  # 100 um
    p_box = p_step * 1.20  # 20% overlap
    offset = (p_box - p_step) / 2.0

    core_w = n_cols * p_step
    core_h = n_rows * p_step

    conn_pitch_um = 9525.0  # 9.525 mm (0.375") SMA spacing
    launch_len_um = 2500.0  # 2.5 mm
    bend_len_um = 5000.0    # 5.0 mm
    board_h_um = conn_pitch_um + 4000.0  # 13.525 mm
    x_p1 = -launch_len_um
    x_b_end = core_w + bend_len_um
    x_p23 = x_b_end + launch_len_um

    # 1. Build Layer 1 Polygons (Top Signal Metal ONLY)
    pixel_boxes = []
    for r in range(n_rows):
        yc = (n_rows / 2.0 - 0.5 - r) * p_step
        for c in range(n_cols):
            if grid[r, c] == 1:
                xc = c * p_step
                b = box(xc - offset, yc - p_step / 2.0 - offset,
                        xc + p_step + offset, yc + p_step / 2.0 + offset)
                pixel_boxes.append(b)

    core_metal = unary_union(pixel_boxes)

    # Input 50-Ohm trace
    p1_trace = box(x_p1, -w_50_um / 2.0, 0.0, w_50_um / 2.0)

    # Symmetrical length-matched S-bends
    y2_start = (n_rows / 4.0) * p_step
    y3_start = -y2_start
    y2_end = conn_pitch_um / 2.0
    y3_end = -y2_end

    x_bend = np.linspace(core_w, x_b_end, 80)
    tau = (x_bend - core_w) / bend_len_um
    y_shift = 0.5 * (1.0 - np.cos(np.pi * tau))
    y2_c = y2_start + (y2_end - y2_start) * y_shift
    y3_c = -y2_c

    def make_ribbon(xc, yc, w):
        top_pts = list(zip(xc, yc + w / 2.0))
        bot_pts = list(zip(xc[::-1], yc[::-1] - w / 2.0))
        return Polygon(top_pts + bot_pts)

    arm2_bend = make_ribbon(x_bend, y2_c, w_50_um)
    arm3_bend = make_ribbon(x_bend, y3_c, w_50_um)

    # Output straight launches
    arm2_launch = box(x_b_end, y2_end - w_50_um / 2.0, x_p23, y2_end + w_50_um / 2.0)
    arm3_launch = box(x_b_end, y3_end - w_50_um / 2.0, x_p23, y3_end + w_50_um / 2.0)

    top_copper_union = unary_union([core_metal, p1_trace, arm2_bend, arm3_bend, arm2_launch, arm3_launch])

    # 2. Build Layer 2 Geometry (Continuous Solid Ground Plane)
    bottom_gnd_poly = box(x_p1, -board_h_um / 2.0, x_p23, board_h_um / 2.0)

    # 3. Create 2-Panel Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), dpi=180)

    # -------------------------------------------------------------
    # Panel 1: Layer 1 (Top Metal - Signal & Pixelated Balun)
    # -------------------------------------------------------------
    # Dielectric background
    sub_patch1 = Rectangle(
        (x_p1 / 1000.0, -board_h_um / 2000.0),
        (x_p23 - x_p1) / 1000.0, board_h_um / 1000.0,
        facecolor='#f8fafc', edgecolor='#94a3b8', linestyle='-', lw=1.2, label='Rogers RO4350B Substrate'
    )
    ax1.add_patch(sub_patch1)

    def draw_poly_on_ax(ax, geom, fc, ec, lbl):
        if isinstance(geom, Polygon):
            ax.add_patch(MplPolygon(np.array(geom.exterior.coords) / 1000.0,
                                    facecolor=fc, edgecolor=ec, lw=1.0, label=lbl))
            for interior in geom.interiors:
                ax.add_patch(MplPolygon(np.array(interior.coords) / 1000.0,
                                        facecolor='#f8fafc', edgecolor=ec, lw=0.8))
        elif isinstance(geom, MultiPolygon):
            for i, p in enumerate(geom.geoms):
                draw_poly_on_ax(ax, p, fc, ec, lbl if i == 0 else "")

    draw_poly_on_ax(ax1, top_copper_union, '#d97706', '#92400e', 'Layer 1: Top Copper (Signal & Pixels, 35 µm)')

    # Ports
    ax1.plot(x_p1 / 1000.0, 0.0, 'o', color='#1d4ed8', markersize=8, label='Port 1: SMA Center Launch (50 Ω)')
    ax1.plot(x_p23 / 1000.0, y2_end / 1000.0, 'o', color='#15803d', markersize=8, label='Port 2: SMA Center Launch (+)')
    ax1.plot(x_p23 / 1000.0, y3_end / 1000.0, 'o', color='#b91c1c', markersize=8, label='Port 3: SMA Center Launch (-)')

    ax1.annotate("Port 1 In (SMA)", xy=(x_p1 / 1000.0, 0.0), xytext=(x_p1 / 1000.0 + 0.3, 1.2),
                 fontsize=8.5, fontweight='bold', color='#1d4ed8',
                 arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.2))
    ax1.annotate("Port 2 Out (+)", xy=(x_p23 / 1000.0, y2_end / 1000.0), xytext=(x_p23 / 1000.0 - 1.8, y2_end / 1000.0 + 1.0),
                 fontsize=8.5, fontweight='bold', color='#15803d',
                 arrowprops=dict(arrowstyle="->", color='#15803d', lw=1.2))
    ax1.annotate("Port 3 Out (-)", xy=(x_p23 / 1000.0, y3_end / 1000.0), xytext=(x_p23 / 1000.0 - 1.8, y3_end / 1000.0 - 1.2),
                 fontsize=8.5, fontweight='bold', color='#b91c1c',
                 arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.2))

    ax1.annotate("Pixelated Balun Core\n(18x18 Discretized Grid)", xy=(core_w / 2000.0, 0.0), xytext=(core_w / 2000.0, -3.2),
                 fontsize=8.5, fontweight='bold', color='#78350f', ha='center',
                 arrowprops=dict(arrowstyle="->", color='#78350f', lw=1.2))

    ax1.set_aspect('equal')
    ax1.set_xlim(x_p1 / 1000.0 - 0.8, x_p23 / 1000.0 + 0.8)
    ax1.set_ylim(-board_h_um / 2000.0 - 0.8, board_h_um / 2000.0 + 0.8)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_ylabel("Y (mm)", fontsize=9.5)
    ax1.set_title("LAYER 1 (TOP): Microstrip Signal Traces & Pixelated Balun Core", fontsize=10.5, fontweight='bold', color='#1e3a8a')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Layer 2 (Bottom Metal - Continuous Solid Ground Plane)
    # -------------------------------------------------------------
    gnd_patch2 = Rectangle(
        (x_p1 / 1000.0, -board_h_um / 2000.0),
        (x_p23 - x_p1) / 1000.0, board_h_um / 1000.0,
        facecolor='#94a3b8', edgecolor='#334155', hatch='//', lw=1.5,
        label='Layer 2: Continuous Solid Copper Ground Plane (35 µm / 1 oz)'
    )
    ax2.add_patch(gnd_patch2)

    # Annotate SMA bottom mounting ground leg solder regions
    sma_tab_w = 2.0
    sma_tab_h = 4.0
    tab1 = Rectangle((x_p1 / 1000.0, -sma_tab_h / 2.0), sma_tab_w, sma_tab_h,
                     facecolor='#cbd5e1', edgecolor='#0f172a', lw=1.2, linestyle='--')
    tab2 = Rectangle((x_p23 / 1000.0 - sma_tab_w, y2_end / 1000.0 - sma_tab_h / 2.0), sma_tab_w, sma_tab_h,
                     facecolor='#cbd5e1', edgecolor='#0f172a', lw=1.2, linestyle='--')
    tab3 = Rectangle((x_p23 / 1000.0 - sma_tab_w, y3_end / 1000.0 - sma_tab_h / 2.0), sma_tab_w, sma_tab_h,
                     facecolor='#cbd5e1', edgecolor='#0f172a', lw=1.2, linestyle='--')
    for t in [tab1, tab2, tab3]:
        ax2.add_patch(t)

    ax2.text(x_p1 / 1000.0 + 1.0, 0.0, "SMA Port 1\nBottom GND Tab",
             ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0f172a')
    ax2.text(x_p23 / 1000.0 - 1.0, y2_end / 1000.0, "SMA Port 2\nBottom GND Tab",
             ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0f172a')
    ax2.text(x_p23 / 1000.0 - 1.0, y3_end / 1000.0, "SMA Port 3\nBottom GND Tab",
             ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0f172a')

    ax2.text((x_p1 + x_p23) / 2000.0, 0.0,
             "CONTINUOUS SOLID COPPER GROUND PLANE (LAYER 2)\n"
             "• Full low-impedance RF ground return across 1 GHz – 40 GHz\n"
             "• Solid ground under all microstrip feeds and pixelated balun core\n"
             "• Solder contact for edge-launch SMA connector bottom ground legs",
             ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='#475569', alpha=0.9))

    ax2.set_aspect('equal')
    ax2.set_xlim(x_p1 / 1000.0 - 0.8, x_p23 / 1000.0 + 0.8)
    ax2.set_ylim(-board_h_um / 2000.0 - 0.8, board_h_um / 2000.0 + 0.8)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_xlabel("X (mm)", fontsize=9.5)
    ax2.set_ylabel("Y (mm)", fontsize=9.5)
    ax2.set_title("LAYER 2 (BOTTOM): Continuous Solid Ground Plane (GND)", fontsize=10.5, fontweight='bold', color='#0f172a')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Layout Figure] Saved 2-layer layout breakdown to {output_path}")


if __name__ == "__main__":
    generate_stackup_diagram("pcb_stackup_2layer.png")
    # Load or generate grid
    grid = np.ones((18, 18), dtype=int)
    # Check if optimized grid exists or generate symmetrical sample
    grid[0:3, 5:13] = 0
    grid[15:18, 5:13] = 0
    grid[4:7, 8:10] = 0
    grid[11:14, 8:10] = 0
    generate_clean_2layer_layout(grid, "pixelated_balun_2layer_layout.png")
