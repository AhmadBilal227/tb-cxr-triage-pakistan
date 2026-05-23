/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional local-dev BYOK seeding (see .env.local). Empty in production. */
  readonly VITE_OPENAI_KEY?: string;
  readonly VITE_HF_TOKEN?: string;
  readonly VITE_REPLICATE_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
