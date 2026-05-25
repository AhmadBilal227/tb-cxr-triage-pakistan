/**
 * clinicianReportPdf — render a radiology-report-style PDF from the v2
 * gpt-interpreter output, the embedded X-ray, and the validated pipeline's
 * audit pins. Intended for the radiologist reviewer; the technical / audit
 * data lives in a small footer rather than the body.
 *
 * Layout (single page, A4 portrait, mm units):
 *   Header band            verdict color-coded, title + date
 *   Two-column body:
 *     - left: text (TECHNIQUE, COMPARISON, FINDINGS by region)
 *     - right: image (75mm wide, aspect-preserved)
 *   Full-width below:      IMPRESSION (numbered, ranked) → RECOMMENDATION → LIMITATIONS
 *   Footer:                research-only disclaimer + audit (model, calibration, sha)
 *
 * Design rules:
 *   - Body text stays in 11pt serif-ish helvetica (jsPDF default — closest to a
 *     real radiology report typographically).
 *   - Section headings are 10pt bold uppercase, the RSNA convention.
 *   - Verdict band is the only chromatic element; the rest is grayscale
 *     so the printed page reads cleanly.
 *   - Audit footer uses 7pt monospace (Courier) so the constants line up.
 *
 * Pagination: if findings + impression overflow, jsPDF auto-paginates only
 * for explicit lines; we manually break to a second page when the cursor
 * crosses the page bottom margin, carrying section headings forward.
 */
import { jsPDF } from 'jspdf';
import type { Adjudication, Verdict } from '@/lib/types';
import type { LocalTriageResult } from '@/lib/providers/localTriage';
import type { ClinicianReport, ImpressionItem } from '@/lib/providers/gptInterpreter';
import { GPT_INTERPRETER_SCHEMA_VERSION } from '@/lib/providers/gptInterpreter';
import { VERDICT_RGB } from '@/lib/colors';

export interface PdfOpts {
  report: ClinicianReport;
  adjudication: Adjudication;
  localResult: LocalTriageResult;
  imageDataUrl: string;
  modelId: string | null;
  latencyMs: number | null;
}

const VERDICT_LABEL: Record<Verdict, string> = {
  tb: 'TB SUSPECTED',
  no_tb: 'NO TB',
  abstain: 'UNCERTAIN — REFER',
};

// VERDICT_RGB (jsPDF wants 0-255 components) is single-sourced from src/lib/colors.

const PAGE_W = 210;
const PAGE_H = 297;
const MARGIN_X = 16;
const MARGIN_BOTTOM = 18;

export function downloadClinicianReportPdf(opts: PdfOpts): void {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const { report, adjudication, localResult, imageDataUrl, modelId, latencyMs } = opts;
  const verdict = adjudication.verdict;

  // --------------------------------------------------------------------
  // Header band — verdict color, title, date.
  // --------------------------------------------------------------------
  const [r, g, b] = VERDICT_RGB[verdict];
  doc.setFillColor(r, g, b);
  doc.rect(0, 0, PAGE_W, 18, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.text('Chest Radiograph Report  ·  TB Triage', MARGIN_X, 8);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(11);
  doc.text(VERDICT_LABEL[verdict], MARGIN_X, 14);

  const generated = new Date().toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
  doc.setFontSize(9);
  const rightLabel = `Generated ${generated}`;
  doc.text(rightLabel, PAGE_W - MARGIN_X - doc.getTextWidth(rightLabel), 14);

  // Reset to body text style.
  doc.setTextColor(20, 20, 20);
  doc.setFont('helvetica', 'normal');

  // --------------------------------------------------------------------
  // Headline — the LLM summary, full width under the verdict band.
  // --------------------------------------------------------------------
  let y = 25;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(20, 20, 20);
  for (const line of doc.splitTextToSize(report.headline, PAGE_W - 2 * MARGIN_X)) {
    doc.text(line, MARGIN_X, y);
    y += 5;
  }
  doc.setFont('helvetica', 'normal');
  y += 3;
  const imageTop = y;

  // --------------------------------------------------------------------
  // Two-column body: text left, image right.
  // --------------------------------------------------------------------
  const COL_LEFT_X = MARGIN_X;
  const COL_RIGHT_X = 122;
  const COL_LEFT_W = COL_RIGHT_X - MARGIN_X - 4;
  const COL_RIGHT_W = PAGE_W - COL_RIGHT_X - MARGIN_X;

  // Image on the right, aspect-preserved. Read the real pixel dimensions
  // via getImageProperties so a non-square CXR is not distorted; clamp the
  // drawn height so a tall portrait film can't overrun the body region.
  let drawnImageH = COL_RIGHT_W; // fallback to square if properties fail
  try {
    const fmt = imageDataUrl.startsWith('data:image/jpeg') ? 'JPEG' : 'PNG';
    const props = doc.getImageProperties(imageDataUrl);
    const aspect = props.height > 0 && props.width > 0 ? props.height / props.width : 1;
    drawnImageH = Math.min(COL_RIGHT_W * aspect, COL_RIGHT_W * 1.4);
    doc.addImage(imageDataUrl, fmt, COL_RIGHT_X, imageTop, COL_RIGHT_W, drawnImageH);
  } catch {
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text('(image embed unavailable)', COL_RIGHT_X, imageTop + 6);
    doc.setTextColor(20, 20, 20);
    drawnImageH = 8;
  }

  // Left column: TECHNIQUE + COMPARISON + IMAGE QUALITY + FINDINGS sections.
  y = sectionHeading(doc, 'Technique', COL_LEFT_X, y);
  y = paragraph(doc, report.technique, COL_LEFT_X, y, COL_LEFT_W);
  y += 2;

  y = sectionHeading(doc, 'Comparison', COL_LEFT_X, y);
  y = paragraph(doc, report.comparison, COL_LEFT_X, y, COL_LEFT_W);
  y += 2;

  y = sectionHeading(doc, 'Image quality', COL_LEFT_X, y);
  y = paragraph(doc, report.image_quality, COL_LEFT_X, y, COL_LEFT_W);
  y += 2;

  y = sectionHeading(doc, 'Findings', COL_LEFT_X, y);
  y = subFinding(doc, 'Lungs and airways', report.findings.lungs_and_airways, COL_LEFT_X, y, COL_LEFT_W);
  y = subFinding(doc, 'Pleura', report.findings.pleura, COL_LEFT_X, y, COL_LEFT_W);
  y = subFinding(doc, 'Cardiomediastinum', report.findings.cardiomediastinum, COL_LEFT_X, y, COL_LEFT_W);
  y = subFinding(doc, 'Bones and soft tissues', report.findings.bones_and_soft_tissues, COL_LEFT_X, y, COL_LEFT_W);

  // Pull cursor below whichever column ended lower so the next full-width
  // section does not overlap the image.
  const yAfterImage = imageTop + drawnImageH + 6;
  y = Math.max(y, yAfterImage);

  // --------------------------------------------------------------------
  // Full-width below: IMPRESSION → RECOMMENDATION → LIMITATIONS.
  // --------------------------------------------------------------------
  const FULL_W = PAGE_W - 2 * MARGIN_X;
  y = sectionHeading(doc, 'Impression', MARGIN_X, y);
  y = renderImpression(doc, report.impression, MARGIN_X, y, FULL_W);
  y += 2;

  y = sectionHeading(doc, 'Recommendation', MARGIN_X, y);
  y = paragraph(doc, report.recommendation, MARGIN_X, y, FULL_W);
  y += 2;

  if (report.support_devices.length > 0) {
    y = sectionHeading(doc, 'Support devices', MARGIN_X, y);
    for (const d of report.support_devices) y = bullet(doc, d, MARGIN_X, y, FULL_W);
    y += 2;
  }

  if (report.incidental_findings.length > 0) {
    y = sectionHeading(doc, 'Incidental findings (non-TB)', MARGIN_X, y);
    for (const f of report.incidental_findings) y = bullet(doc, f, MARGIN_X, y, FULL_W);
    y += 2;
  }

  if (report.limitations.length > 0) {
    y = sectionHeading(doc, 'Limitations', MARGIN_X, y);
    for (const l of report.limitations) {
      y = bullet(doc, l, MARGIN_X, y, FULL_W);
    }
  }

  // --------------------------------------------------------------------
  // Footer — research-only disclaimer + audit (model, calibration, sha).
  // Fits on the last page; if body overflows we already paginated above.
  // --------------------------------------------------------------------
  drawFooter(doc, adjudication, localResult, modelId, latencyMs);

  doc.save(filenameFor(verdict));
}

// ----------------------------------------------------------------------
// Layout primitives
// ----------------------------------------------------------------------

function pageBreakIfNeeded(doc: jsPDF, y: number, lineHeight = 6): number {
  if (y > PAGE_H - MARGIN_BOTTOM - lineHeight) {
    doc.addPage();
    return 20;
  }
  return y;
}

function sectionHeading(doc: jsPDF, label: string, x: number, y: number): number {
  y = pageBreakIfNeeded(doc, y, 8);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(60, 60, 60);
  doc.text(label.toUpperCase(), x, y);
  doc.setTextColor(20, 20, 20);
  doc.setFont('helvetica', 'normal');
  // Underline
  const lw = doc.getTextWidth(label.toUpperCase());
  doc.setDrawColor(180, 180, 180);
  doc.line(x, y + 1, x + lw, y + 1);
  return y + 5;
}

function paragraph(doc: jsPDF, text: string, x: number, y: number, w: number): number {
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  const lines = doc.splitTextToSize(text || '—', w);
  for (const line of lines) {
    y = pageBreakIfNeeded(doc, y);
    doc.text(line, x, y);
    y += 4.5;
  }
  return y;
}

function subFinding(doc: jsPDF, label: string, text: string, x: number, y: number, w: number): number {
  y = pageBreakIfNeeded(doc, y, 10);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.text(`${label}:`, x, y);
  doc.setFont('helvetica', 'normal');
  const labelW = doc.getTextWidth(`${label}: `);
  doc.setFontSize(10);
  const lines = doc.splitTextToSize(text || '—', w - labelW);
  // First line sits next to the bold label; subsequent lines wrap to x.
  if (lines.length > 0) {
    doc.text(lines[0], x + labelW, y);
    y += 4.5;
    for (let i = 1; i < lines.length; i++) {
      y = pageBreakIfNeeded(doc, y);
      doc.text(lines[i], x, y);
      y += 4.5;
    }
  }
  return y + 1;
}

function renderImpression(
  doc: jsPDF,
  items: ImpressionItem[],
  x: number,
  y: number,
  w: number,
): number {
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  items.forEach((item, i) => {
    y = pageBreakIfNeeded(doc, y, 10);
    const numLabel = `${i + 1}.`;
    doc.setFont('helvetica', 'bold');
    doc.text(numLabel, x, y);
    doc.setFont('helvetica', 'normal');
    const numW = doc.getTextWidth(`${numLabel} `);
    const tag =
      item.likelihood === 'primary'
        ? ''
        : item.likelihood === 'consider'
          ? '  [consider]'
          : '  [less likely]';
    const text = item.statement + tag;
    const lines = doc.splitTextToSize(text, w - numW);
    doc.text(lines[0], x + numW, y);
    y += 5;
    for (let j = 1; j < lines.length; j++) {
      y = pageBreakIfNeeded(doc, y);
      doc.text(lines[j], x + numW, y);
      y += 5;
    }
  });
  return y;
}

function bullet(doc: jsPDF, text: string, x: number, y: number, w: number): number {
  y = pageBreakIfNeeded(doc, y);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.text('•', x, y);
  const lines = doc.splitTextToSize(text, w - 4);
  doc.text(lines[0], x + 4, y);
  y += 4;
  for (let j = 1; j < lines.length; j++) {
    y = pageBreakIfNeeded(doc, y);
    doc.text(lines[j], x + 4, y);
    y += 4;
  }
  return y + 0.5;
}

function drawFooter(
  doc: jsPDF,
  adjudication: Adjudication,
  local: LocalTriageResult,
  modelId: string | null,
  latencyMs: number | null,
): void {
  const totalPages = doc.getNumberOfPages();
  doc.setPage(totalPages);
  const yBase = PAGE_H - MARGIN_BOTTOM + 2;

  // Separator line.
  doc.setDrawColor(200, 200, 200);
  doc.line(MARGIN_X, yBase - 4, PAGE_W - MARGIN_X, yBase - 4);

  doc.setFont('helvetica', 'italic');
  doc.setFontSize(8);
  doc.setTextColor(80, 80, 80);
  const disclaimer =
    'Research preview — not a medical device. AI-assisted radiographic screen. Bacteriological confirmation required for diagnosis.';
  const dLines = doc.splitTextToSize(disclaimer, PAGE_W - 2 * MARGIN_X);
  let y = yBase;
  for (const line of dLines) {
    doc.text(line, MARGIN_X, y);
    y += 3.4;
  }

  // Audit pins — monospace so the constants line up.
  doc.setFont('courier', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(110, 110, 110);
  const cal = local.audit.calibration;
  const auditLines = [
    `perception_path: ${adjudication.perception_path ?? 'unknown'} · interpreter_schema: ${GPT_INTERPRETER_SCHEMA_VERSION} · narrative_model: ${modelId ?? 'gpt-5.5'} · latency: ${latencyMs ? (latencyMs / 1000).toFixed(1) + 's' : 'n/a'}`,
    `tb_prob: ${local.tb_prob.toFixed(4)} · s_inactive: ${local.s_inactive.toFixed(4)} · decided_at: ${local.decided_at_threshold.toFixed(4)} · T: ${cal.T.toFixed(3)} · T_seq: ${cal.T_sequelae.toFixed(3)}`,
    `model_sha: ${local.audit.model_sha.slice(0, 16)} · git_sha: ${local.audit.git_sha.slice(0, 7)} · model_id: ${local.audit.model_id} · ts: ${local.audit.timestamp}`,
  ];
  for (const line of auditLines) {
    doc.text(line, MARGIN_X, y);
    y += 3;
  }
  doc.setTextColor(20, 20, 20);
}

function filenameFor(verdict: Verdict): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  return `tb-triage-report-${verdict}-${stamp}.pdf`;
}
