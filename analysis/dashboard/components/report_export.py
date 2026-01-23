"""
Clinical Report Export Utilities

Provides functions to export clinical reports in various formats:
- PDF (professional formatting)
- HTML (with styling)
- Markdown
- JSON
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import io

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def generate_html_report(
    individual_name: str,
    report: Any,
    trajectory: List[str],
    classification: Dict = None,
    data_source: str = "actual"
) -> str:
    """
    Generate a styled HTML clinical report.

    Args:
        individual_name: Name of the individual
        report: ClinicalReport object from intervention module
        trajectory: List of states
        classification: Classification data dict
        data_source: Whether data is "actual" or "estimated"

    Returns:
        HTML string
    """
    # Get report components
    risk = report.risk_assessment
    monitoring = report.monitoring_plan
    recommendations = report.recommended_interventions
    missed = report.missed_opportunities if hasattr(report, 'missed_opportunities') else []

    # Risk level color mapping
    risk_colors = {
        'Low': '#27ae60',
        'Moderate': '#f39c12',
        'High': '#e67e22',
        'Critical': '#c0392b'
    }

    risk_level = risk.level.value if risk else 'Unknown'
    risk_color = risk_colors.get(risk_level, '#95a5a6')

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Clinical Report: {individual_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            color: #2c3e50;
        }}
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
            background-color: {risk_color};
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .section h2 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .metric {{
            display: inline-block;
            padding: 15px 25px;
            margin: 5px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            font-size: 0.85em;
            color: #7f8c8d;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }}
        .recommendation {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .recommendation h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 0.85em;
        }}
        .trajectory-box {{
            font-family: monospace;
            background: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: nowrap;
        }}
        .state-seeking {{ color: #3498db; }}
        .state-directing {{ color: #e74c3c; }}
        .state-conferring {{ color: #27ae60; }}
        .state-revising {{ color: #f39c12; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Clinical Intervention Report</h1>
        <p class="subtitle">Individual: <strong>{individual_name.replace('_', ' ')}</strong></p>
        <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p class="subtitle">Data source: {data_source.title()}</p>
    </div>

    <div style="text-align: center; margin: 30px 0;">
        <span class="risk-badge">Risk Level: {risk_level}</span>
    </div>

    <div style="display: flex; justify-content: center; flex-wrap: wrap; margin: 20px 0;">
        <div class="metric">
            <div class="metric-value">{len(trajectory)}</div>
            <div class="metric-label">Events</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report.mfpt_to_directing:.1f}</div>
            <div class="metric-label">MFPT to Directing</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(report.intervention_windows)}</div>
            <div class="metric-label">Intervention Windows</div>
        </div>
    </div>
"""

    # Risk Assessment Section
    if risk:
        html += f"""
    <div class="section">
        <h2>Risk Assessment</h2>
        <p><strong>Overall Score:</strong> {risk.score:.2f}</p>
        <p><strong>Trajectory Indicators:</strong></p>
        <ul>
"""
        for indicator in risk.trajectory_indicators:
            html += f"            <li>{indicator}</li>\n"
        html += """        </ul>
    </div>
"""

    # Trajectory Visualization
    state_spans = []
    for s in trajectory[-20:]:  # Show last 20 states
        state_class = f"state-{s.lower()}"
        state_spans.append(f'<span class="{state_class}">{s}</span>')

    html += f"""
    <div class="section">
        <h2>Behavioral Trajectory (Last 20 Events)</h2>
        <div class="trajectory-box">
            {' → '.join(state_spans)}
        </div>
    </div>
"""

    # Recommended Interventions
    if recommendations:
        html += """
    <div class="section">
        <h2>Recommended Interventions</h2>
"""
        for i, rec in enumerate(recommendations[:5], 1):
            protocol = rec.protocol
            html += f"""
        <div class="recommendation">
            <h4>{i}. {protocol.display_name}</h4>
            <table>
                <tr><td><strong>Expected Benefit:</strong></td><td>{rec.expected_benefit:.1%}</td></tr>
                <tr><td><strong>Cost:</strong></td><td>${rec.cost:,.0f}</td></tr>
                <tr><td><strong>Urgency:</strong></td><td>{rec.urgency:.1%}</td></tr>
                <tr><td><strong>Timing:</strong></td><td>Event {rec.time_index}</td></tr>
            </table>
            <p><em>{rec.rationale}</em></p>
        </div>
"""
        html += "    </div>\n"

    # Monitoring Plan
    if monitoring:
        html += f"""
    <div class="section">
        <h2>Monitoring Plan</h2>
        <p><strong>Frequency:</strong> {monitoring.frequency}</p>
        <p><strong>Reassessment Timeline:</strong> {monitoring.reassessment_timeline}</p>

        <h3>Key Indicators</h3>
        <ul>
"""
        for ind in monitoring.key_indicators:
            html += f"            <li>{ind}</li>\n"

        html += """        </ul>

        <h3>Escalation Triggers</h3>
"""
        for trigger in monitoring.escalation_triggers:
            html += f'        <div class="warning">{trigger}</div>\n'

        html += "    </div>\n"

    # Classification info if available
    if classification:
        primary = classification.get('primary_type', 'Unknown')
        subtype = classification.get('subtype', 'Unknown')
        html += f"""
    <div class="section">
        <h2>Classification</h2>
        <p><strong>Primary Type:</strong> {primary}</p>
        <p><strong>Subtype:</strong> {subtype}</p>
    </div>
"""

    # Footer
    html += """
    <div class="footer">
        <p><strong>Disclaimer:</strong> This report is for clinical decision support only.
        All recommendations require professional clinical judgment and should not be used
        as the sole basis for clinical decisions.</p>
        <p>Generated by the Clinical Intervention Analysis System</p>
    </div>
</body>
</html>
"""
    return html


def report_to_html_download(report: Any, individual_name: str, trajectory: List[str],
                            classification: Dict = None, data_source: str = "actual") -> tuple:
    """
    Generate HTML report for download.

    Returns:
        Tuple of (html_content, filename, mimetype)
    """
    html = generate_html_report(individual_name, report, trajectory, classification, data_source)
    filename = f"clinical_report_{individual_name}.html"
    return html, filename, "text/html"


def _create_pdf_class():
    """Create PDF class only if FPDF is available."""
    if not PDF_AVAILABLE:
        return None

    class ClinicalReportPDF(FPDF):
        """Custom PDF class for clinical reports."""

        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)

        def header(self):
            self.set_font('Helvetica', 'B', 12)
            self.set_text_color(44, 62, 80)
            self.cell(0, 10, 'Clinical Intervention Report', align='C', new_x='LMARGIN', new_y='NEXT')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

        def section_title(self, title: str):
            self.set_x(10)  # Ensure left margin
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(52, 152, 219)
            self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
            self.set_draw_color(52, 152, 219)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)
            self.set_x(10)  # Reset X after line

        def body_text(self, text: str):
            self.set_x(10)  # Ensure left margin
            self.set_font('Helvetica', '', 10)
            self.set_text_color(51, 51, 51)
            self.multi_cell(0, 6, text)
            self.ln(2)

        def metric_box(self, label: str, value: str, x: float, y: float, width: float = 50):
            self.set_xy(x, y)
            self.set_fill_color(248, 249, 250)
            self.set_draw_color(200, 200, 200)
            self.rect(x, y, width, 20, style='DF')
            self.set_xy(x, y + 3)
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(52, 152, 219)
            self.cell(width, 8, value, align='C')
            self.set_xy(x, y + 12)
            self.set_font('Helvetica', '', 8)
            self.set_text_color(128, 128, 128)
            self.cell(width, 6, label, align='C')

        def risk_badge(self, level: str, score: float):
            colors = {
                'Low': (39, 174, 96),
                'Moderate': (243, 156, 18),
                'High': (230, 126, 34),
                'Critical': (192, 57, 43)
            }
            r, g, b = colors.get(level, (149, 165, 166))
            self.set_fill_color(r, g, b)
            self.set_text_color(255, 255, 255)
            self.set_font('Helvetica', 'B', 12)
            badge_width = 80
            x = (210 - badge_width) / 2
            self.set_xy(x, self.get_y())
            self.cell(badge_width, 12, f'Risk Level: {level}', align='C', fill=True)
            self.ln(15)

    return ClinicalReportPDF


def generate_pdf_report(
    individual_name: str,
    report: Any,
    trajectory: List[str],
    classification: Dict = None,
    data_source: str = "actual"
) -> bytes:
    """
    Generate a professional PDF clinical report.

    Args:
        individual_name: Name of the individual
        report: ClinicalReport object from intervention module
        trajectory: List of states
        classification: Classification data dict
        data_source: Whether data is "actual" or "estimated"

    Returns:
        PDF bytes
    """
    if not PDF_AVAILABLE:
        raise ImportError("fpdf2 is required for PDF generation. Install with: pip install fpdf2")

    ClinicalReportPDF = _create_pdf_class()
    pdf = ClinicalReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Get report components
    risk = report.risk_assessment
    monitoring = report.monitoring_plan
    recommendations = report.recommended_interventions

    # Title and metadata
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, 'Clinical Intervention Report', align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 6, f'Individual: {individual_name.replace("_", " ")}', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, f'Data Source: {data_source.title()}', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(10)

    # Risk Badge
    if risk:
        pdf.risk_badge(risk.level.value, risk.score)

    # Key Metrics
    y_pos = pdf.get_y()
    pdf.metric_box('Events', str(len(trajectory)), 25, y_pos, 50)
    pdf.metric_box('MFPT', f'{report.mfpt_to_directing:.1f}', 80, y_pos, 50)
    pdf.metric_box('Windows', str(len(report.intervention_windows)), 135, y_pos, 50)
    pdf.set_xy(10, y_pos + 30)  # Reset both X and Y position

    # Risk Assessment Section
    if risk:
        pdf.section_title('Risk Assessment')
        pdf.body_text(f'Overall Score: {risk.score:.2f}')
        pdf.ln(3)
        pdf.set_x(10)  # Ensure left margin
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Trajectory Indicators:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 10)
        for indicator in risk.trajectory_indicators:
            pdf.set_x(10)  # Reset X for each line
            pdf.multi_cell(0, 5, f'  * {indicator}')
        pdf.ln(5)

    # Behavioral Trajectory
    pdf.section_title('Behavioral Trajectory')
    last_states = trajectory[-15:]  # Show last 15 states
    trajectory_text = ' -> '.join(last_states)
    pdf.set_x(10)
    pdf.set_font('Courier', '', 9)
    pdf.set_text_color(51, 51, 51)
    pdf.multi_cell(0, 5, f'Last {len(last_states)} events: {trajectory_text}')
    pdf.ln(5)

    # Recommended Interventions
    if recommendations:
        pdf.section_title('Recommended Interventions')
        for i, rec in enumerate(recommendations[:5], 1):
            protocol = rec.protocol
            pdf.set_x(10)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 8, f'{i}. {protocol.display_name}', new_x='LMARGIN', new_y='NEXT')

            pdf.set_x(10)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(51, 51, 51)
            pdf.cell(0, 6, f'  Expected Benefit: {rec.expected_benefit:.1%}  |  Cost: ${rec.cost:,.0f}  |  Urgency: {rec.urgency:.1%}', new_x='LMARGIN', new_y='NEXT')

            if rec.rationale:
                pdf.set_x(10)
                pdf.set_font('Helvetica', 'I', 9)
                pdf.set_text_color(128, 128, 128)
                pdf.multi_cell(0, 5, '  ' + rec.rationale[:150] + ('...' if len(rec.rationale) > 150 else ''))
            pdf.ln(3)

    # Monitoring Plan
    if monitoring:
        pdf.add_page()
        pdf.section_title('Monitoring Plan')
        pdf.body_text(f'Frequency: {monitoring.frequency}')
        pdf.body_text(f'Reassessment Timeline: {monitoring.reassessment_timeline}')
        pdf.ln(3)

        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Key Indicators:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 10)
        for ind in monitoring.key_indicators:
            pdf.set_x(10)
            pdf.multi_cell(0, 5, f'  * {ind}')
        pdf.ln(3)

        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Escalation Triggers:', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(192, 57, 43)
        for trigger in monitoring.escalation_triggers:
            pdf.set_x(10)
            pdf.multi_cell(0, 5, f'  ! {trigger}')
        pdf.ln(5)

    # Classification
    if classification:
        pdf.set_text_color(51, 51, 51)
        pdf.section_title('Classification')
        pdf.body_text(f'Primary Type: {classification.get("primary_type", "Unknown")}')
        pdf.body_text(f'Subtype: {classification.get("subtype", "Unknown")}')

    # Disclaimer
    pdf.ln(10)
    pdf.set_x(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4,
        'Disclaimer: This report is for clinical decision support only. '
        'All recommendations require professional clinical judgment and should not be used '
        'as the sole basis for clinical decisions.'
    )

    # Return PDF bytes
    return pdf.output()


def is_pdf_available() -> bool:
    """Check if PDF generation is available."""
    return PDF_AVAILABLE
