import io
import html
from datetime import datetime
from typing import List, Dict, Optional, Any
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that calculates total pages and prints running headers,
    footers, page numbers ('Page X of Y'), and official verification watermarks.
    """
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
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        
        # ── Top Running Header ──
        self.setStrokeColor(colors.HexColor("#94a3b8"))
        self.setLineWidth(0.5)
        self.line(20, 815, 575, 815)
        
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1e3a8a"))
        self.drawString(20, 821, "TAMIL NADU ENGINEERING ADMISSIONS (TNEA) 2026")
        
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(575, 821, "DIRECTORATE OF TECHNICAL EDUCATION (DoTE) • OFFICIAL DIRECTORY")
        
        # ── Bottom Running Footer ──
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(20, 30, 575, 30)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(20, 19, "Official TNEA Counseling Data • Directorate of Technical Education, Chennai 600 025")
        
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1e293b"))
        footer_right = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(575, 19, footer_right)
        
        self.restoreState()


def generate_colleges_pdf(
    colleges: List[Dict[str, Any]],
    branch_code: Optional[str] = None,
    branch_name: Optional[str] = None,
    district: Optional[str] = None,
    total_seats: Optional[int] = None
) -> bytes:
    """
    Generates an executive, highly structured, multi-page official PDF catalog
    of Tamil Nadu Engineering Colleges offering the specified branch/district.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20,
        leftMargin=20,
        topMargin=38,
        bottomMargin=42
    )

    elements = []
    styles = getSampleStyleSheet()

    # ── Custom Typography Styles ──
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0f2942'),
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#475569'),
        spaceAfter=8
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1e293b')
    )

    cell_center_style = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=1, # Centered
        textColor=colors.HexColor('#334155')
    )

    cell_code_style = ParagraphStyle(
        'TableCellCode',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=1, # Centered
        textColor=colors.HexColor('#1e40af')
    )

    cell_intake_style = ParagraphStyle(
        'TableCellIntake',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=2, # Right aligned
        textColor=colors.HexColor('#0f172a')
    )

    header_cell_center = ParagraphStyle(
        'HeaderCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.white
    )

    header_cell_left = ParagraphStyle(
        'HeaderCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=0,
        textColor=colors.white
    )

    header_cell_right = ParagraphStyle(
        'HeaderCellRight',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=2,
        textColor=colors.white
    )

    # ── 1. Document Masthead ──
    elements.append(Paragraph("TNEA 2026 — OFFICIAL INSTITUTIONAL SEAT MATRIX", title_style))
    elements.append(Paragraph("Directorate of Technical Education (DoTE) • Government of Tamil Nadu", subtitle_style))

    # ── 2. Executive Summary Metrics Box ──
    course_display = branch_name or (f"Branch Code: {branch_code}" if branch_code else "All Engineering Programs")
    dist_display = district.title() if district else "Statewide (All 38 Districts)"
    seats_display = f"{total_seats:,}" if total_seats is not None else "Refer Table"
    
    summary_data = [
        [
            Paragraph("<b>Course / Program:</b>", cell_style),
            Paragraph(html.escape(course_display), cell_style),
            Paragraph("<b>Region / Scope:</b>", cell_style),
            Paragraph(html.escape(dist_display), cell_style),
        ],
        [
            Paragraph("<b>TNEA Branch Code:</b>", cell_style),
            Paragraph(f"<b>{html.escape(branch_code or 'All')}</b>", cell_code_style),
            Paragraph("<b>Total Institutions:</b>", cell_style),
            Paragraph(f"<b>{len(colleges):,} Colleges</b>", cell_style),
        ],
        [
            Paragraph("<b>Total Approved Intake:</b>", cell_style),
            Paragraph(f"<b>{seats_display} Seats</b>", cell_intake_style),
            Paragraph("<b>Counselling Year:</b>", cell_style),
            Paragraph("<b>2026 – 2027 Academic Session</b>", cell_style),
        ]
    ]

    summary_table = Table(summary_data, colWidths=[110, 155, 110, 155])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 8))

    # ── 3. Main Data Table ──
    has_specific_branch = bool(branch_code)
    seats_header = f"{branch_code} Intake" if has_specific_branch else "Intake"

    table_data = [[
        Paragraph("<b>S.No</b>", header_cell_center),
        Paragraph("<b>Code</b>", header_cell_center),
        Paragraph("<b>College Name & Campus Location</b>", header_cell_left),
        Paragraph("<b>District</b>", header_cell_left),
        Paragraph("<b>Status</b>", header_cell_center),
        Paragraph(f"<b>{seats_header}</b>", header_cell_right)
    ]]

    # Populate rows
    for i, item in enumerate(colleges):
        meta = item["metadata"] if "metadata" in item else item

        code = html.escape(str(meta.get("tnea_code") or meta.get("code") or "N/A")).strip()
        raw_name = str(meta.get("college_name") or meta.get("name") or "Unknown College").strip()
        
        # Split primary name from full street address
        parts = [p.strip() for p in raw_name.split(",") if p.strip()]
        if len(parts) > 1:
            if "University Departments of Anna University" in parts[0] and len(parts) > 1:
                primary_name = f"{parts[0]}, {parts[1]}"
                address = ", ".join(parts[2:])
            else:
                primary_name = parts[0]
                address = ", ".join(parts[1:])
            
            safe_primary = html.escape(primary_name)
            safe_address = html.escape(address)
            name_cell_html = f"<b>{safe_primary}</b>"
            if safe_address:
                name_cell_html += f"<br/><font color='#64748b' size='6.2'>{safe_address}</font>"
        else:
            name_cell_html = f"<b>{html.escape(raw_name)}</b>"
        
        dist = html.escape(str(meta.get("district") or "Tamil Nadu")).strip()
        is_auto = bool(meta.get("autonomous", False))
        status_html = "<font color='#047857'><b>Autonomous</b></font>" if is_auto else "<font color='#64748b'>Affiliated</font>"
        
        # Extract seats
        if has_specific_branch:
            intakes = meta.get("department_intakes", {})
            if isinstance(intakes, dict) and branch_code in intakes:
                seats = str(intakes[branch_code])
            elif "intake" in meta:
                seats = str(meta["intake"])
            else:
                seats = "—"
        else:
            seats = str(meta.get("total_intake") or "—")

        table_data.append([
            Paragraph(str(i + 1), cell_center_style),
            Paragraph(code, cell_code_style),
            Paragraph(name_cell_html, cell_style),
            Paragraph(dist, cell_style),
            Paragraph(status_html, cell_center_style),
            Paragraph(html.escape(seats), cell_intake_style)
        ])

    col_widths = [26, 38, 274, 76, 68, 48]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f2942')), # Deep executive navy
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#0f2942')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))

    elements.append(t)

    # Build PDF with dynamic footer
    doc.build(elements, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
