"""
generate_zoomed_core_figure.py - Generates a detailed zoom-in view of the 18x18 pixelated balun core.
Highlights:
 1. 100 µm pixel grid cells and 20% overlap region for DC/RF galvanic continuity.
 2. Port 1 single-ended microstrip feed interface (left).
 3. Port 2 (+) and Port 3 (-) balanced differential feed interfaces (right).
 4. Symmetry plane along y = 0 and dimension callouts.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

from pcb_stackup import AVAILABLE_SUBSTRATES


def generate_zoomed_core_figure(
    grid: np.ndarray,
    tech = None,
    pixel_size_um: float = 100.0,
    overlap_ratio: float = 0.20,
    output_path: str = "pixelated_core_zoomed.png"
):
    if tech is None:
        tech = AVAILABLE_SUBSTRATES["Rogers_RO4350B"]

    w_50_m, _ = tech.synth_microstrip_width(50.0)
    w_50_um = w_50_m * 1e6  # ~368 um

    n_rows, n_cols = grid.shape
    p_step = pixel_size_um
    p_box = p_step * (1.0 + overlap_ratio)
    offset = (p_box - p_step) / 2.0

    core_w = n_cols * p_step
    core_h = n_rows * p_step

    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)

    # 1. Background substrate area
    sub_bg = Rectangle(
        (-600.0 / 1000.0, -core_h / 2000.0 - 0.25),
        (core_w + 1200.0) / 1000.0, (core_h + 500.0) / 1000.0,
        facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.2, label='Rogers RO4350B Substrate'
    )
    ax.add_patch(sub_bg)

    # 2. Draw underlying 18x18 grid cell boundaries (dashed gray lines)
    for c in range(n_cols + 1):
        x = (c * p_step) / 1000.0
        ax.plot([x, x], [-core_h / 2000.0, core_h / 2000.0],
                color='#cbd5e1', linestyle=':', lw=0.7, zorder=2)
    for r in range(n_rows + 1):
        y = (core_h / 2.0 - r * p_step) / 1000.0
        ax.plot([0.0, core_w / 1000.0], [y, y],
                color='#cbd5e1', linestyle=':', lw=0.7, zorder=2)

    # 3. Draw individual metal pixel boxes showing 20% overlap
    pixel_boxes = []
    for r in range(n_rows):
        yc = (n_rows / 2.0 - 0.5 - r) * p_step
        for c in range(n_cols):
            if grid[r, c] == 1:
                xc = c * p_step
                # Physical overlapping box
                b = box(xc - offset, yc - p_step / 2.0 - offset,
                        xc + p_step + offset, yc + p_step / 2.0 + offset)
                pixel_boxes.append(b)

                # Draw individual pixel outline to highlight overlap
                p_rect = Rectangle(
                    ((xc - offset) / 1000.0, (yc - p_step / 2.0 - offset) / 1000.0),
                    p_box / 1000.0, p_box / 1000.0,
                    facecolor='#d97706', edgecolor='#b45309', lw=0.6, alpha=0.75, zorder=3
                )
                ax.add_patch(p_rect)

    # 4. Port 1 Input Feed Line (x: -500 um to 0, y: 0)
    feed_in = Rectangle(
        (-500.0 / 1000.0, -w_50_um / 2000.0),
        500.0 / 1000.0, w_50_um / 1000.0,
        facecolor='#d97706', edgecolor='#92400e', lw=1.2, zorder=4,
        label='50 Ω Microstrip Feed (W = 0.368 mm)'
    )
    ax.add_patch(feed_in)

    # 5. Differential Output Feed Lines (Ports 2 & 3)
    y2_c = (n_rows / 4.0) * p_step
    y3_c = -y2_c
    feed_out2 = Rectangle(
        (core_w / 1000.0, (y2_c - w_50_um / 2.0) / 1000.0),
        500.0 / 1000.0, w_50_um / 1000.0,
        facecolor='#d97706', edgecolor='#92400e', lw=1.2, zorder=4
    )
    feed_out3 = Rectangle(
        (core_w / 1000.0, (y3_c - w_50_um / 2.0) / 1000.0),
        500.0 / 1000.0, w_50_um / 1000.0,
        facecolor='#d97706', edgecolor='#92400e', lw=1.2, zorder=4
    )
    ax.add_patch(feed_out2)
    ax.add_patch(feed_out3)

    # 6. Symmetry plane along y = 0
    ax.axhline(0.0, color='#3b82f6', linestyle='--', lw=1.2, alpha=0.7, zorder=5, label='Symmetry Plane (y = 0)')

    # 7. Port Markers
    ax.plot(-0.5, 0.0, 'o', color='#1d4ed8', markersize=9, zorder=10, label='Port 1: Single-Ended Input')
    ax.plot((core_w + 500.0) / 1000.0, y2_c / 1000.0, 'o', color='#15803d', markersize=9, zorder=10, label='Port 2: Balanced Output (+)')
    ax.plot((core_w + 500.0) / 1000.0, y3_c / 1000.0, 'o', color='#b91c1c', markersize=9, zorder=10, label='Port 3: Balanced Output (-)')

    # 8. Annotations & Dimension Callouts
    # Port 1
    ax.annotate("Port 1 In\n(50 Ω Unbalanced)",
                xy=(-0.5, 0.0), xytext=(-0.55, 0.45),
                fontsize=9, fontweight='bold', color='#1d4ed8', ha='center',
                arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.4))

    # Port 2
    ax.annotate("Port 2 Out (+)\n(50 Ω Balanced)",
                xy=((core_w + 500.0) / 1000.0, y2_c / 1000.0),
                xytext=((core_w + 500.0) / 1000.0, y2_c / 1000.0 + 0.35),
                fontsize=9, fontweight='bold', color='#15803d', ha='center',
                arrowprops=dict(arrowstyle="->", color='#15803d', lw=1.4))

    # Port 3
    ax.annotate("Port 3 Out (-)\n(50 Ω Balanced)",
                xy=((core_w + 500.0) / 1000.0, y3_c / 1000.0),
                xytext=((core_w + 500.0) / 1000.0, y3_c / 1000.0 - 0.45),
                fontsize=9, fontweight='bold', color='#b91c1c', ha='center',
                arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.4))

    # Pixel Size Callout
    sample_c, sample_r = 2, 2
    sx = (sample_c * p_step) / 1000.0
    sy = (core_h / 2.0 - sample_r * p_step) / 1000.0
    ax.annotate(
        "Unit Pixel:\n100 µm × 100 µm Grid\n(120 µm with 20% Overlap)",
        xy=(sx + 0.05, sy - 0.05), xytext=(sx + 0.35, sy + 0.45),
        fontsize=8.5, fontweight='bold', color='#78350f',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='#fef3c7', edgecolor='#d97706', lw=1.0),
        arrowprops=dict(arrowstyle="->", color='#d97706', lw=1.4)
    )

    # Core Dimensions
    ax.annotate("", xy=(0.0, -core_h / 2000.0 - 0.12), xytext=(core_w / 1000.0, -core_h / 2000.0 - 0.12),
                arrowprops=dict(arrowstyle="<->", color='#334155', lw=1.2))
    ax.text(core_w / 2000.0, -core_h / 2000.0 - 0.18, f"Core Width: {core_w/1000:.2f} mm (18 Pixels)",
            ha='center', va='top', fontsize=8.5, fontweight='bold', color='#334155')

    # Shift height arrow to the left edge of substrate
    ax.annotate("", xy=(-0.58, -core_h / 2000.0), xytext=(-0.58, core_h / 2000.0),
                arrowprops=dict(arrowstyle="<->", color='#334155', lw=1.2))
    ax.text(-0.64, 0.0, f"Core Height: {core_h/1000:.2f} mm (18 Pixels)",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#334155', rotation=90)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-0.7, core_w / 1000.0 + 0.7)
    ax.set_ylim(-core_h / 2000.0 - 0.35, core_h / 2000.0 + 0.35)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlabel("X Position (mm)", fontsize=10)
    ax.set_ylabel("Y Position (mm)", fontsize=10)
    ax.set_title("Zoomed-In View: 18×18 Pixelated Balun Core Architecture", fontsize=12, fontweight='bold', pad=12)

    # Deduplicate legend items
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[Zoomed Core] Saved zoomed-in core view to {output_path}")


if __name__ == "__main__":
    # Create symmetrical grid matching current optimized balun
    grid = np.ones((18, 18), dtype=int)
    grid[0:3, 5:13] = 0
    grid[15:18, 5:13] = 0
    grid[4:7, 8:10] = 0
    grid[11:14, 8:10] = 0
    grid[7:11, 4:7] = 0
    generate_zoomed_core_figure(grid, output_path="pixelated_core_zoomed.png")
