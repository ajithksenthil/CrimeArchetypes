"""
Clinical Report Export Utilities

Provides functions to export clinical reports in various formats:
- HTML (with styling)
- Markdown
- JSON
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


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
