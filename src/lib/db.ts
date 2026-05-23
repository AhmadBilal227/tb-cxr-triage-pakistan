import Dexie, { type Table } from 'dexie';
import type { CaseHistoryRecord, LabeledCaseRecord } from './types';
import { shortId } from './utils';

/**
 * Client-side persistence. No backend, no DB server — IndexedDB via Dexie.
 *
 * - labeledCases: the RAG corpus. Stores the image blob + its embedding + label.
 *   Grows over time via labeled-set import and the verdict-card "Disagree?" flow.
 * - caseHistory: thumbnails + full run records for the left-rail history.
 */
class TbTriageDB extends Dexie {
  labeledCases!: Table<LabeledCaseRecord, string>;
  caseHistory!: Table<CaseHistoryRecord, string>;

  constructor() {
    super('tb-triage');
    this.version(1).stores({
      // Index label + source for fast corpus stats; embedding stays unindexed (large array).
      labeledCases: 'id, filename, label, source, createdAt',
      caseHistory: 'id, createdAt, verdict',
    });
  }
}

export const db = new TbTriageDB();

// ---------------------------------------------------------------------------
// Labeled corpus (RAG)
// ---------------------------------------------------------------------------

export async function addLabeledCase(
  rec: Omit<LabeledCaseRecord, 'id' | 'createdAt'> & { id?: string },
): Promise<string> {
  const id = rec.id ?? shortId();
  await db.labeledCases.put({ ...rec, id, createdAt: Date.now() });
  return id;
}

/** All labeled cases that have a usable embedding (kNN candidates). */
export async function getEmbeddedCases(): Promise<LabeledCaseRecord[]> {
  const all = await db.labeledCases.toArray();
  return all.filter((c) => Array.isArray(c.embedding) && c.embedding.length > 0);
}

export async function countLabeledCases(): Promise<{
  total: number;
  embedded: number;
  tb: number;
  notTb: number;
}> {
  const all = await db.labeledCases.toArray();
  return {
    total: all.length,
    embedded: all.filter((c) => c.embedding && c.embedding.length > 0).length,
    tb: all.filter((c) => c.label === 1).length,
    notTb: all.filter((c) => c.label === 0).length,
  };
}

export async function clearLabeledCases(): Promise<void> {
  await db.labeledCases.clear();
}

// ---------------------------------------------------------------------------
// Case history (left rail)
// ---------------------------------------------------------------------------

export async function addHistory(rec: CaseHistoryRecord): Promise<void> {
  await db.caseHistory.put(rec);
}

export async function listHistory(limit = 50): Promise<CaseHistoryRecord[]> {
  return db.caseHistory.orderBy('createdAt').reverse().limit(limit).toArray();
}

export async function getHistory(id: string): Promise<CaseHistoryRecord | undefined> {
  return db.caseHistory.get(id);
}

export async function deleteHistory(id: string): Promise<void> {
  await db.caseHistory.delete(id);
}
