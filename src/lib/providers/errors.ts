import type { Provider } from '@/lib/types';

/** Base class for any provider-level failure. Carries enough context for the trace UI. */
export class ProviderError extends Error {
  readonly provider: Provider;
  readonly status?: number;
  readonly raw?: unknown;

  constructor(message: string, provider: Provider, status?: number, raw?: unknown) {
    super(message);
    this.name = 'ProviderError';
    this.provider = provider;
    this.status = status;
    this.raw = raw;
  }
}

export class HfError extends ProviderError {
  constructor(message: string, status?: number, raw?: unknown) {
    super(message, 'hf', status, raw);
    this.name = 'HfError';
  }
}

export class ReplicateError extends ProviderError {
  constructor(message: string, status?: number, raw?: unknown) {
    super(message, 'replicate', status, raw);
    this.name = 'ReplicateError';
  }
}

export class OpenAIError extends ProviderError {
  constructor(message: string, status?: number, raw?: unknown) {
    super(message, 'openai', status, raw);
    this.name = 'OpenAIError';
  }
}

/** Thrown when a stage cannot run because the user has not configured it. */
export class NotConfiguredError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NotConfiguredError';
  }
}
