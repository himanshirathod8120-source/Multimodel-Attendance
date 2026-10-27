"""
session_analytics.py
====================
Generates per-student and class-wide session analytics.
Creates engagement timeline charts, heatmaps, and summary tables.
Exports HTML and PDF reports.
"""

import time
import json
import logging
import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SessionAnalytics:
    """
    Aggregates all session data and generates reports.
    """

    def __init__(self, config: dict):
        self.report_dir = Path(config.get("output_dir", "data/reports"))
        self.generate_pdf = config.get("generate_pdf", True)
        self.generate_html = config.get("generate_html", True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._session_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Chart generation helpers
    # ------------------------------------------------------------------

    def _engagement_timeline_chart(self, snapshots: list[dict]) -> str:
        """Generate engagement timeline as base64 PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 4), facecolor="#1a1a2e")
            ax.set_facecolor("#16213e")

            # Group by student
            students = list({s["student_id"] for s in snapshots})
            colors = plt.cm.Set2(np.linspace(0, 1, max(len(students), 1)))

            for student, color in zip(students, colors):
                data = [(s["timestamp"], s["engagement_score"])
                        for s in snapshots if s["student_id"] == student]
                if not data:
                    continue
                times, scores = zip(*data)
                ax.plot(times, scores, label=student, color=color, linewidth=2, alpha=0.85)
                ax.fill_between(times, scores, alpha=0.15, color=color)

            ax.axhline(y=40, color="#ff6b6b", linestyle="--", alpha=0.5, linewidth=1, label="Alert threshold")
            ax.set_xlabel("Session Time (seconds)", color="white", fontsize=10)
            ax.set_ylabel("Engagement Score", color="white", fontsize=10)
            ax.set_title("Engagement Timeline", color="white", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 100)
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#2a2a4a")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Could not generate timeline chart: {e}")
            return ""

    def _score_bar_chart(self, averages: dict[str, float]) -> str:
        """Generate student average engagement bar chart as base64 PNG."""
        if not averages:
            return ""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = list(averages.keys())
            scores = list(averages.values())
            colors = ["#4CAF50" if s >= 70 else "#FFC107" if s >= 40 else "#F44336" for s in scores]

            fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4), facecolor="#1a1a2e")
            ax.set_facecolor("#16213e")
            bars = ax.bar(names, scores, color=colors, edgecolor="#2a2a4a", linewidth=0.5)

            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{score:.0f}", ha="center", va="bottom", color="white", fontsize=9)

            ax.set_ylim(0, 110)
            ax.set_ylabel("Avg Engagement Score", color="white")
            ax.set_title("Student Engagement Summary", color="white", fontsize=13, fontweight="bold")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#2a2a4a")
            plt.xticks(rotation=20, ha="right", color="white")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Could not generate bar chart: {e}")
            return ""

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        attendance_summary: dict,
        engagement_snapshots: list[dict],
        student_averages: dict[str, float],
        audio_summary: dict,
        suggestions: list[str],
        session_duration_s: float,
    ) -> Path:
        """Generate and save HTML report."""
        timeline_img = self._engagement_timeline_chart(engagement_snapshots)
        bar_img = self._score_bar_chart(student_averages)
        class_avg = round(float(np.mean(list(student_averages.values()))) if student_averages else 0, 1)

        rows = ""
        for name, score in sorted(student_averages.items(), key=lambda x: -x[1]):
            status = "✅ Present" if name in [p["student_id"] for p in attendance_summary.get("present", [])] else "❌ Absent"
            color = "#4CAF50" if score >= 70 else "#FFC107" if score >= 40 else "#F44336"
            rows += f"""
                <tr>
                    <td>{name}</td>
                    <td style="color:{color};font-weight:bold">{score:.0f}</td>
                    <td>{status}</td>
                </tr>"""

        suggestions_html = "".join(f"<li>{s}</li>" for s in suggestions) if suggestions else "<li>No suggestions generated.</li>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session Report — {self._session_date}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', sans-serif; background: #0f0f1a; color: #e0e0f0; padding: 2rem; }}
  h1 {{ font-size: 2rem; font-weight: 700; color: #7c6af0; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.2rem; color: #a0a0c0; font-weight: 400; margin-bottom: 2rem; }}
  h3 {{ font-size: 1rem; color: #7c6af0; margin-bottom: 1rem; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 1.5rem; text-align: center; }}
  .card .val {{ font-size: 2.2rem; font-weight: 700; color: #7c6af0; }}
  .card .label {{ font-size: 0.8rem; color: #7070a0; margin-top: 0.3rem; }}
  .section {{ background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  img {{ max-width: 100%; border-radius: 8px; margin-top: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #2a2a4a; color: #a0a0c0; padding: 0.75rem; text-align: left; }}
  td {{ padding: 0.75rem; border-bottom: 1px solid #2a2a4a; }}
  tr:last-child td {{ border-bottom: none; }}
  ul.suggestions {{ list-style: none; padding: 0; }}
  ul.suggestions li {{ background: #0f0f2a; border-left: 3px solid #7c6af0; padding: 0.75rem 1rem; margin-bottom: 0.75rem; border-radius: 6px; font-size: 0.9rem; line-height: 1.5; }}
  .footer {{ text-align: center; color: #404060; font-size: 0.75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>🎓 Multimodal Attendance & Engagement Report</h1>
<h2>Session: {self._session_date} &nbsp;|&nbsp; Duration: {session_duration_s/60:.1f} min</h2>

<div class="grid">
  <div class="card"><div class="val">{attendance_summary.get('present_count', 0)}</div><div class="label">Students Present</div></div>
  <div class="card"><div class="val">{class_avg}</div><div class="label">Class Engagement Score</div></div>
  <div class="card"><div class="val">{audio_summary.get('speech_percentage', 0):.0f}%</div><div class="label">Speech Activity</div></div>
  <div class="card"><div class="val">{len(suggestions)}</div><div class="label">AI Suggestions Given</div></div>
</div>

<div class="section">
  <h3>📈 Engagement Timeline</h3>
  {'<img src="data:image/png;base64,' + timeline_img + '" alt="Engagement Timeline">' if timeline_img else '<p>No data.</p>'}
</div>

<div class="section">
  <h3>📊 Student Engagement Summary</h3>
  {'<img src="data:image/png;base64,' + bar_img + '" alt="Engagement Bars">' if bar_img else ''}
  <table style="margin-top:1rem">
    <thead><tr><th>Student</th><th>Avg Engagement</th><th>Attendance</th></tr></thead>
    <tbody>{rows if rows else '<tr><td colspan="3">No students tracked.</td></tr>'}</tbody>
  </table>
</div>

<div class="section">
  <h3>🤖 AI Teaching Suggestions</h3>
  <ul class="suggestions">{suggestions_html}</ul>
</div>

<div class="section">
  <h3>🎤 Audio Summary</h3>
  <table>
    <tr><td>Total Frames Analyzed</td><td>{audio_summary.get('total_frames_analyzed', 0)}</td></tr>
    <tr><td>Speech Activity</td><td>{audio_summary.get('speech_percentage', 0):.1f}%</td></tr>
    <tr><td>Avg Audio Score</td><td>{audio_summary.get('avg_audio_score', 0):.3f}</td></tr>
  </table>
</div>

<div class="footer">Generated by Multimodal Attendance &amp; Engagement Agent &bull; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body>
</html>"""

        out_path = self.report_dir / f"report_{self._session_date}.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report → {out_path}")
        return out_path

    def generate_pdf_report(self, html_path: Path) -> Optional[Path]:
        """Generate PDF from session data using reportlab."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.lib.units import inch

            pdf_path = html_path.with_suffix(".pdf")
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle("title", parent=styles["Title"],
                                          textColor=colors.HexColor("#7c6af0"), fontSize=20)
            story.append(Paragraph("Multimodal Attendance & Engagement Report", title_style))
            story.append(Paragraph(f"Session: {self._session_date}", styles["Normal"]))
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("Report generated. See HTML version for full charts.", styles["Normal"]))

            doc.build(story)
            logger.info(f"PDF report → {pdf_path}")
            return pdf_path
        except ImportError:
            logger.warning("reportlab not installed. Skipping PDF generation.")
            return None
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None
