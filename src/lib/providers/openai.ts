import { OpenAIError } from './errors';
import { classifyHttpFailure, providerStatusStore } from '@/store/providerStatus';

/**
 * OpenAI — ORCHESTRATION ONLY (quality gate, VLM ensemble member, adjudicator).
 * Uses the Responses API (/v1/responses) with gpt-5.5 vision.
 *
 * GPT-5.5 has no Replicate fallback; on failure we drop to gpt-5.5-instant.
 */

const OPENAI_BASE = 'https://api.openai.com/v1';

export interface JsonSchemaFormat {
  name: string;
  schema: Record<string, unknown>;
}

interface ResponsesBody {
  model: string;
  input: unknown;
  text?: { format: { type: 'json_object' } | { type: 'json_schema'; name: string; schema: Record<string, unknown>; strict: boolean } };
  stream?: boolean;
}

function buildVisionInput(prompt: string, imageDataUrl: string): unknown {
  return [
    {
      role: 'user',
      content: [
        { type: 'input_text', text: prompt },
        { type: 'input_image', image_url: imageDataUrl },
      ],
    },
  ];
}

function buildTextInput(prompt: string): unknown {
  return [{ role: 'user', content: [{ type: 'input_text', text: prompt }] }];
}

/** Walk the Responses API output array to concatenate assistant text. */
function extractOutputText(json: unknown): string {
  const j = json as {
    output_text?: string;
    output?: Array<{ type?: string; content?: Array<{ type?: string; text?: string }> }>;
  };
  if (typeof j.output_text === 'string' && j.output_text.length > 0) return j.output_text;
  let text = '';
  for (const item of j.output ?? []) {
    if (item.type === 'message' && Array.isArray(item.content)) {
      for (const c of item.content) {
        if (c.type === 'output_text' && typeof c.text === 'string') text += c.text;
      }
    }
  }
  return text;
}

/** Tolerant JSON extraction: strips code fences, slices the outermost object. */
export function extractJSON<T>(text: string): T {
  const cleaned = text.replace(/```json/gi, '').replace(/```/g, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end === -1 || end < start) {
    throw new OpenAIError(`could not find JSON in model output: ${text.slice(0, 120)}`);
  }
  try {
    return JSON.parse(cleaned.slice(start, end + 1)) as T;
  } catch (err) {
    throw new OpenAIError(`invalid JSON from model: ${(err as Error).message}`);
  }
}

async function postResponses(apiKey: string, body: ResponsesBody): Promise<Response> {
  return fetch(`${OPENAI_BASE}/responses`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
}

export interface VisionJsonOpts {
  apiKey: string;
  model: string;
  fallbackModel: string;
  prompt: string;
  /** Omit for text-only adjudication context that still wants vision? Provide for vision. */
  imageDataUrl?: string;
  schema?: JsonSchemaFormat;
}

export interface OpenAIJsonResult<T> {
  data: T;
  raw: unknown;
  modelUsed: string;
  latencyMs: number;
  fellBack: boolean;
}

/** Non-streaming JSON call with automatic gpt-5.5 -> gpt-5.5-instant fallback. */
export async function openaiJSON<T>(opts: VisionJsonOpts): Promise<OpenAIJsonResult<T>> {
  if (!opts.apiKey) {
    providerStatusStore.set('openai', { state: 'not-configured', note: 'no API key' });
    throw new OpenAIError('OpenAI API key missing');
  }

  const models = [opts.model, opts.fallbackModel].filter(
    (m, i, a) => m && a.indexOf(m) === i,
  );
  let lastErr: Error | null = null;

  for (let i = 0; i < models.length; i++) {
    const model = models[i] as string;
    const start = performance.now();
    try {
      const body: ResponsesBody = {
        model,
        input: opts.imageDataUrl
          ? buildVisionInput(opts.prompt, opts.imageDataUrl)
          : buildTextInput(opts.prompt),
        text: opts.schema
          ? { format: { type: 'json_schema', name: opts.schema.name, schema: opts.schema.schema, strict: true } }
          : { format: { type: 'json_object' } },
      };
      const res = await postResponses(opts.apiKey, body);
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        const classified = classifyHttpFailure(res.status, text);
        providerStatusStore.set('openai', {
          state: classified.state,
          note: `${classified.humanReason} (model ${model})`,
        });
        throw new OpenAIError(`OpenAI ${res.status}: ${text.slice(0, 240)}`, res.status, text);
      }
      const raw: unknown = await res.json();
      const data = extractJSON<T>(extractOutputText(raw));
      providerStatusStore.set('openai', { state: 'ok', note: `model ${model}` });
      return { data, raw, modelUsed: model, latencyMs: performance.now() - start, fellBack: i > 0 };
    } catch (err) {
      lastErr = err as Error;
      // try next (fallback) model
    }
  }
  throw lastErr ?? new OpenAIError('OpenAI call failed');
}

export interface StreamOpts {
  apiKey: string;
  model: string;
  fallbackModel: string;
  prompt: string;
  imageDataUrl?: string;
  schema?: JsonSchemaFormat;
  onToken: (token: string) => void;
}

export interface StreamResult {
  text: string;
  modelUsed: string;
  latencyMs: number;
  fellBack: boolean;
}

/**
 * Streaming Responses call. Emits output text deltas via onToken and returns the
 * full accumulated text. Falls back to the instant model if the primary errors
 * before producing any tokens.
 */
export async function openaiStream(opts: StreamOpts): Promise<StreamResult> {
  if (!opts.apiKey) {
    providerStatusStore.set('openai', { state: 'not-configured', note: 'no API key' });
    throw new OpenAIError('OpenAI API key missing');
  }
  const models = [opts.model, opts.fallbackModel].filter(
    (m, i, a) => m && a.indexOf(m) === i,
  );
  let lastErr: Error | null = null;

  for (let i = 0; i < models.length; i++) {
    const model = models[i] as string;
    const start = performance.now();
    try {
      const body: ResponsesBody = {
        model,
        input: opts.imageDataUrl
          ? buildVisionInput(opts.prompt, opts.imageDataUrl)
          : buildTextInput(opts.prompt),
        text: opts.schema
          ? { format: { type: 'json_schema', name: opts.schema.name, schema: opts.schema.schema, strict: true } }
          : { format: { type: 'json_object' } },
        stream: true,
      };
      const res = await postResponses(opts.apiKey, body);
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => '');
        const classified = classifyHttpFailure(res.status, text);
        providerStatusStore.set('openai', {
          state: classified.state,
          note: `stream: ${classified.humanReason} (model ${model})`,
        });
        throw new OpenAIError(`OpenAI ${res.status}: ${text.slice(0, 240)}`, res.status, text);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let full = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() ?? '';
        for (const block of blocks) {
          for (const line of block.split('\n')) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const payload = trimmed.slice(5).trim();
            if (payload === '[DONE]') continue;
            try {
              const evt = JSON.parse(payload) as { type?: string; delta?: string };
              if (evt.type === 'response.output_text.delta' && typeof evt.delta === 'string') {
                full += evt.delta;
                opts.onToken(evt.delta);
              }
            } catch {
              // ignore keep-alive / partial frames
            }
          }
        }
      }
      providerStatusStore.set('openai', { state: 'ok', note: `stream model ${model}` });
      return { text: full, modelUsed: model, latencyMs: performance.now() - start, fellBack: i > 0 };
    } catch (err) {
      lastErr = err as Error;
    }
  }
  throw lastErr ?? new OpenAIError('OpenAI stream failed');
}
