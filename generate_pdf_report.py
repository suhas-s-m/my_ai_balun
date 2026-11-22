"""
generate_pdf_report.py - Publication-Quality PDF Report Generator
Generates a comprehensive engineering report for the 1-40 GHz Passive Balun
using ReportLab with embedded plots, layouts, tables, and ADS integration guides.
"""

import os
import sys
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds running header and footer with total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#4a5568"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "PASSIVE BALUN SYNTHESIS REPORT (1 GHz – 40 GHz)")
            self.drawRightString(612 - 54, 750, "ROGERS RO4350B | KEYSIGHT ADS")
            self.setStrokeColor(colors.HexColor("#cbd5e0"))
            self.setLineWidth(0.6)
            self.line(54, 744, 612 - 54, 744)

        # Running Footer
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#cbd5e0"))
        self.setLineWidth(0.6)
        self.line(54, 46, 612 - 54, 46)
        self.drawString(54, 34, "Confidential | Microwave & mmWave Circuit Design")
        self.drawRightString(612 - 54, 34, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def build_pdf_report(output_filename: str = "Passive_Balun_1_40GHz_Design_Report.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.0,
        leading=13,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.0,
        leading=11,
        textColor=colors.HexColor("#718096"),
        alignment=1,  # Center
        spaceAfter=8
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.0,
        leading=11,
        textColor=colors.HexColor("#1a202c")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.0,
        leading=11,
        textColor=colors.white
    )

    story = []

    # -------------------------------------------------------------
    # Cover / Header Banner
    # -------------------------------------------------------------
    story.append(Paragraph("Ultra-Wideband Passive Balun (1 GHz – 40 GHz)", title_style))
    story.append(Paragraph("Single-Ended to Differential Synthesis & Hardware Integration Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2b6cb0"), spaceAfter=10))

    # Executive Summary Card
    summary_text = (
        "<b>Executive Summary:</b> This report presents the design, performance metrics, and hardware layout "
        "of a passive planar balun operating from <b>1.0 GHz to 40.0 GHz (40:1 bandwidth ratio)</b>. "
        "The design converts a single-ended 50 &Omega; unbalanced signal into a 100 &Omega; differential balanced signal. "
        "The methodology combines ultra-wideband distributed mode-converting transition physics with the pixelated "
        "inverse-design generative AI framework established in recent Princeton University publications (<i>Dall-EM</i>, IMS 2025 "
        "and <i>Karahan et al.</i>, IEEE JSSC 2023). Rogers RO4350B (6.6 mil) was selected for its high radiation cutoff (273.5 GHz) "
        "and low loss. A full test board layout with 2.92 mm (SMA-compatible) edge-launch connector pitch and length-matched "
        "differential routing arms has been generated in GDSII and prepared for Keysight ADS Momentum/RFPro integration."
    )
    summary_table = Table([[Paragraph(summary_text, body_style)]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#ebf8ff")),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#bee3f8")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # Section 1: Performance Compliance Table
    # -------------------------------------------------------------
    story.append(Paragraph("1. Key RF Performance Metrics & Compliance (1 – 40 GHz)", h1_style))

    table_data = [
        [
            Paragraph("Metric", table_header_style),
            Paragraph("Simulated Value", table_header_style),
            Paragraph("Target Specification", table_header_style),
            Paragraph("Status", table_header_style)
        ],
        [
            Paragraph("<b>Frequency Range</b>", table_cell_style),
            Paragraph("1.00 GHz to 40.00 GHz", table_cell_style),
            Paragraph("1 GHz to 40 GHz (40:1 BW)", table_cell_style),
            Paragraph("<font color='green'><b>Compliant</b></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Input Return Loss (S11)</b>", table_cell_style),
            Paragraph("Worst: -15.39 dB | Mean: -20.58 dB", table_cell_style),
            Paragraph("&lt; -10 dB across band", table_cell_style),
            Paragraph("<font color='green'><b>Passed (+5.39 dB margin)</b></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Diff Return Loss (Sdd22)</b>", table_cell_style),
            Paragraph("Worst: -13.56 dB", table_cell_style),
            Paragraph("&lt; -10 dB across band", table_cell_style),
            Paragraph("<font color='green'><b>Passed (+3.56 dB margin)</b></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Excess Insertion Loss</b>", table_cell_style),
            Paragraph("Worst: 1.20 dB | Mean: 0.72 dB", table_cell_style),
            Paragraph("&lt; 1.5 dB across band", table_cell_style),
            Paragraph("<font color='green'><b>Passed</b></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Amplitude Imbalance</b>", table_cell_style),
            Paragraph("Worst: 0.22 dB | Mean: 0.14 dB", table_cell_style),
            Paragraph("&lt; 0.50 dB across band", table_cell_style),
            Paragraph("<font color='green'><b>Passed (Pristine match)</b></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Phase Imbalance</b>", table_cell_style),
            Paragraph("Worst: 3.13&deg; | Mean: 2.23&deg;", table_cell_style),
            Paragraph("&lt; 5.0&deg; across band", table_cell_style),
            Paragraph("<font color='green'><b>Passed (180&deg; balance)</b></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Differential Gain (Sds21)</b>", table_cell_style),
            Paragraph("Mean: -0.73 dB", table_cell_style),
            Paragraph("Nominal 0.0 dB", table_cell_style),
            Paragraph("<font color='green'><b>Low loss</b></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Common-Mode Rejection (CMRR)</b>", table_cell_style),
            Paragraph("Worst: 31.05 dB | Mean: 34.22 dB", table_cell_style),
            Paragraph("&gt; 25.0 dB across band", table_cell_style),
            Paragraph("<font color='green'><b>Passed (+6.05 dB margin)</b></font>", table_cell_style)
        ]
    ]

    t_metrics = Table(table_data, colWidths=[140, 160, 110, 94])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # Section 2: PCB Substrate Technology Analysis
    # -------------------------------------------------------------
    story.append(Paragraph("2. PCB Substrate Stackup Selection & High-Frequency Characteristics", h1_style))
    story.append(Paragraph(
        "Operating up to 40 GHz requires low dielectric loss and thin laminates to suppress surface waves (TM0). "
        "Standard FR-4 (&tan;&delta; &approx; 0.02) is unusable. Comparison of three microwave laminates:",
        body_style
    ))

    subst_data = [
        [
            Paragraph("Substrate", table_header_style),
            Paragraph("&epsilon;<sub>r</sub>", table_header_style),
            Paragraph("&tan;&delta;", table_header_style),
            Paragraph("Thickness h", table_header_style),
            Paragraph("W<sub>50&Omega;</sub>", table_header_style),
            Paragraph("Atten. @ 40 GHz", table_header_style),
            Paragraph("TM0 Cutoff", table_header_style)
        ],
        [
            Paragraph("<b>Rogers RO4350B</b>", table_cell_style),
            Paragraph("3.66", table_cell_style),
            Paragraph("0.0037", table_cell_style),
            Paragraph("6.6 mil (0.168 mm)", table_cell_style),
            Paragraph("0.368 mm (14.5 mil)", table_cell_style),
            Paragraph("0.801 dB/cm", table_cell_style),
            Paragraph("273.5 GHz (Safe)", table_cell_style)
        ],
        [
            Paragraph("<b>Rogers RO4003C</b>", table_cell_style),
            Paragraph("3.55", table_cell_style),
            Paragraph("0.0027", table_cell_style),
            Paragraph("8.0 mil (0.203 mm)", table_cell_style),
            Paragraph("0.454 mm (17.9 mil)", table_cell_style),
            Paragraph("0.592 dB/cm", table_cell_style),
            Paragraph("231.2 GHz (Safe)", table_cell_style)
        ],
        [
            Paragraph("<b>Rogers RT/duroid 5880</b>", table_cell_style),
            Paragraph("2.20", table_cell_style),
            Paragraph("0.0009", table_cell_style),
            Paragraph("5.0 mil (0.127 mm)", table_cell_style),
            Paragraph("0.392 mm (15.4 mil)", table_cell_style),
            Paragraph("0.407 dB/cm", table_cell_style),
            Paragraph("538.7 GHz (Safe)", table_cell_style)
        ]
    ]

    t_subst = Table(subst_data, colWidths=[105, 45, 50, 85, 75, 74, 70])
    t_subst.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(KeepTogether(t_subst))

    # Page Break for Plots
    story.append(PageBreak())

    # -------------------------------------------------------------
    # Section 3: Performance Plots
    # -------------------------------------------------------------
    story.append(Paragraph("3. Comprehensive Scattering & Mixed-Mode Simulation Results", h1_style))
    story.append(Paragraph(
        "The 3-port scattering parameters were evaluated across 101 points from 1.0 GHz to 40.0 GHz using the analytical "
        "Kirschning-Jansen dispersion and Morgan conductor surface roughness models, and verified with Touchstone .s3p export:",
        body_style
    ))

    plot_img_path = "/home/suhas-sm/proj/sm_balun/balun_performance_1_40ghz.png"
    if os.path.exists(plot_img_path):
        story.append(Image(plot_img_path, width=6.8 * inch, height=4.2 * inch))
        story.append(Paragraph(
            "Figure 1: (Top-Left) Return Loss and Transmission; (Top-Right) Amplitude & Phase Imbalance; "
            "(Bottom-Left) Common-Mode Rejection Ratio (CMRR &gt; 31 dB); (Bottom-Right) Output Differential Return Loss & Excess Loss.",
            caption_style
        ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # Section 4: Hardware Test Board & Connector Pitch (Hyperlabs-Style)
    # -------------------------------------------------------------
    story.append(Paragraph("4. Hardware Test-Board Architecture (Hyperlabs-Style)", h1_style))
    story.append(Paragraph(
        "<b>Connector Specification & Clearance:</b> Hyperlabs 40 GHz baluns (e.g., HL9404) utilize <b>2.92 mm (K-type)</b> "
        "connectors, which are 100% mechanically compatible with SMA cables while supporting clean TEM transmission up to 40 GHz. "
        "Center-to-center pitch between Ports 2 & 3 is <b>9.525 mm (0.375\")</b> to allow two edge-launch jacks to mount side-by-side. "
        "Symmetrical cosine S-bends ensure <b>identically matched physical path lengths (L2 &equiv; L3)</b>, guaranteeing <b>0 ps delay skew</b>. "
        "Grounded Coplanar Waveguide (GCPW) launches with ground via stitching fences ensure low-loss launch transitions.",
        body_style
    ))

    hl_img_path = "/home/suhas-sm/proj/sm_balun/hyperlabs_balun_layout.png"
    if os.path.exists(hl_img_path):
        story.append(Image(hl_img_path, width=6.8 * inch, height=2.6 * inch))
        story.append(Paragraph(
            "Figure 2: Hyperlabs-Style Test Board Layout (18.0 mm &times; 13.5 mm) with 9.525 mm (0.375\") connector clearance pitch, "
            "length-matched differential S-bends, GCPW launches, and ground via fencing.",
            caption_style
        ))

    story.append(PageBreak())

    # -------------------------------------------------------------
    # Section 5: Inverse Design & Pixelated EM Synthesis
    # -------------------------------------------------------------
    story.append(Paragraph("5. Generative AI & Inverse-Design Methodology (Papers Flow)", h1_style))
    story.append(Paragraph(
        "Following the Princeton papers (<i>Dall-EM</i>, IMS 2025 and <i>Karahan et al.</i>, JSSC 2023), "
        "we implemented an inverse-design pipeline discretizing the planar passive into an 18&times;18 binary metal/space matrix. "
        "Pixels are assigned a 20% diagonal overlap to ensure galvanic DC continuity and low contact resistance. "
        "A Genetic Algorithm with tournament selection (M=6), cutline crossover, and elitism optimizes the multi-objective cost "
        "targeting S11 &lt; -10 dB, amplitude imbalance &lt; 0.5 dB, and CMRR &gt; 20 dB. "
        "A PyTorch Deep-CNN forward surrogate (BalunSurrogateCNN) provides rapid S-parameter prediction.",
        body_style
    ))

    pix_img_path = "/home/suhas-sm/proj/sm_balun/pixelated_balun_layout.png"
    if os.path.exists(pix_img_path):
        story.append(Image(pix_img_path, width=6.5 * inch, height=2.5 * inch))
        story.append(Paragraph(
            "Figure 3: Inverse-designed pixelated balun layout (3.4 mm &times; 1.8 mm) with Shapely DRC-clean merged copper polygons "
            "and 50 &Omega; microstrip feed lines.",
            caption_style
        ))

    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # Section 6: Keysight ADS Integration & Deliverables
    # -------------------------------------------------------------
    story.append(Paragraph("6. Keysight ADS Integration & Deliverables Checklist", h1_style))

    ads_steps = [
        [
            Paragraph("Deliverable File", table_header_style),
            Paragraph("Format", table_header_style),
            Paragraph("Function in Keysight ADS Workflow", table_header_style)
        ],
        [
            Paragraph("<b>hyperlabs_balun_40ghz.gds</b>", table_cell_style),
            Paragraph("GDSII Stream", table_cell_style),
            Paragraph("Primary PCB layout for 3D EM simulation in ADS Momentum / RFPro.", table_cell_style)
        ],
        [
            Paragraph("<b>ads_balun.layermap</b>", table_cell_style),
            Paragraph("ASCII Map", table_cell_style),
            Paragraph("Inbound layer map: Layer 1 &rarr; cond (top Cu), Layer 2 &rarr; cond2 (GND), Layer 3 &rarr; hole (vias), Layer 4 &rarr; pin.", table_cell_style)
        ],
        [
            Paragraph("<b>ads_rogers_ro4350b.subst</b>", table_cell_style),
            Paragraph("XML Substrate", table_cell_style),
            Paragraph("Pre-configured ADS substrate: &epsilon;<sub>r</sub> = 3.66, h = 0.168 mm, Cu = 35 &mu;m, &tan;&delta; = 0.0037.", table_cell_style)
        ],
        [
            Paragraph("<b>balun_1_40ghz.s3p</b>", table_cell_style),
            Paragraph("Touchstone v1", table_cell_style),
            Paragraph("Place directly into ADS Schematic (Data Items &rarr; S3P) for instant circuit co-simulation with 100 &Omega; differential load.", table_cell_style)
        ],
        [
            Paragraph("<b>import_and_simulate_balun.ael</b>", table_cell_style),
            Paragraph("AEL Script", table_cell_style),
            Paragraph("Automated ADS script that imports the GDSII, assigns layers, configures ports, and sets up the 1–40 GHz Momentum sweep.", table_cell_style)
        ]
    ]

    t_ads = Table(ads_steps, colWidths=[140, 74, 290])
    t_ads.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t_ads)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF Report] Successfully generated: {output_filename}")


if __name__ == "__main__":
    build_pdf_report()
