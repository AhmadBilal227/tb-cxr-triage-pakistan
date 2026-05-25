/**
 * PathologyList — M24 component test.
 *
 * Pins: 18 TXRV labels render, sorted desc by probability, severity buckets
 * (low/mid/high) correctly assigned, honest "not independent diagnoses"
 * framing always present.
 */
import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { PathologyList } from './PathologyList';
import type { TxrvPathologies } from '@/lib/providers/localTriage';

// Eighteen labels match the M24 server output (xrv.models.DenseNet.pathologies).
const SAMPLE: TxrvPathologies = {
  'Lung Opacity': 0.5183,
  Consolidation: 0.2542,
  Pneumonia: 0.2529,
  Infiltration: 0.2164,
  'Lung Lesion': 0.1645,
  Effusion: 0.1,
  Atelectasis: 0.08,
  Pneumothorax: 0.05,
  Cardiomegaly: 0.04,
  Fibrosis: 0.03,
  Edema: 0.02,
  Mass: 0.018,
  Nodule: 0.015,
  Emphysema: 0.01,
  Pleural_Thickening: 0.008,
  Hernia: 0.005,
  Fracture: 0.003,
  'Enlarged Cardiomediastinum': 0.001,
};

describe('PathologyList', () => {
  it('renders the honest "not independent diagnoses" framing', () => {
    const html = renderToStaticMarkup(<PathologyList pathologies={SAMPLE} />);
    expect(html).toContain('Other findings the perception backbone sees');
    expect(html).toContain('Not independent diagnoses');
  });

  it('renders all 18 supplied chips', () => {
    const html = renderToStaticMarkup(<PathologyList pathologies={SAMPLE} />);
    expect(html).toContain('data-testid="pathology-chip-Lung-Opacity"');
    expect(html).toContain('data-testid="pathology-chip-Consolidation"');
    expect(html).toContain('data-testid="pathology-chip-Enlarged-Cardiomediastinum"');
    // Quick sanity on count — 18 chip divs. The component also emits a wrapper
    // `data-testid="pathology-chip-list"` which would match the broad regex;
    // pin the per-chip count via the trailing-letter shape instead.
    const chipCount = (html.match(/data-testid="pathology-chip-[A-Z]/g) ?? []).length;
    expect(chipCount).toBe(18);
  });

  it('sorts descending — Lung Opacity (0.52) comes before Consolidation (0.25)', () => {
    const html = renderToStaticMarkup(<PathologyList pathologies={SAMPLE} />);
    const idxFirst = html.indexOf('data-testid="pathology-chip-Lung-Opacity"');
    const idxSecond = html.indexOf('data-testid="pathology-chip-Consolidation"');
    expect(idxFirst).toBeGreaterThan(0);
    expect(idxSecond).toBeGreaterThan(idxFirst);
  });

  it('tags severity correctly (high>=0.5, mid 0.2..0.5, low<=0.2)', () => {
    const html = renderToStaticMarkup(<PathologyList pathologies={SAMPLE} />);
    // Lung Opacity 0.52 -> high
    expect(html).toMatch(/data-testid="pathology-chip-Lung-Opacity"[^>]*data-severity="high"/);
    // Consolidation 0.25 -> mid
    expect(html).toMatch(/data-testid="pathology-chip-Consolidation"[^>]*data-severity="mid"/);
    // Edema 0.02 -> low
    expect(html).toMatch(/data-testid="pathology-chip-Edema"[^>]*data-severity="low"/);
  });

  it('renders the empty-state when no scores are supplied', () => {
    const html = renderToStaticMarkup(<PathologyList pathologies={{}} />);
    expect(html).toContain('No pathology scores in this run');
  });
});
