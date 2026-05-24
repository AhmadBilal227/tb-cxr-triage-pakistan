"""Local-mode triage CLI (Milestone 22).

Wraps `triage_core.TriageEngine` so the user can run real-numbers calibrated verdicts
on their own machine, without spinning up the server. Output mode is selectable:

    training/.venv/bin/python training/triage_cli.py <image_path>                 # default --human
    training/.venv/bin/python training/triage_cli.py <image_path> --json          # raw TriageResult JSON
    training/.venv/bin/python training/triage_cli.py <image_path> --include-gpt   # add the gpt-5.5
                                                                                  # vision second opinion

`--include-gpt` calls the OpenAI Responses API with the SAME `submit_triage` schema
that src/lib/pipeline/vlmTriage.ts uses, then prints BOTH numbers side-by-side. The
project's ethos is intellectual honesty about model quality: when the local model
says one thing and a generic VLM says another, the user owes themselves the
disagreement signal, not a "winner". Off by default to keep the CLI offline-clean.

Key (OpenAI) comes from .env.local via VITE_OPENAI_KEY (same source the browser app
already uses), with an OPENAI_API_KEY fallback for non-browser shells. No key, no
GPT call — `--include-gpt` exits with status 2 rather than silently skipping it.

Environment-wise: HF_HUB_OFFLINE=1 is the default so the engine never reaches out
on a warm cache. Set HF_HUB_OFFLINE=0 explicitly to warm a new machine; once cached
it should stay offline forever.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

# Default to offline before triage_core imports transformers; safe to override below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from triage_core import TriageEngine, TriageResult  # noqa: E402


# ---------------------------------------------------------------------------
# Optional GPT verifier
# ---------------------------------------------------------------------------
_ENV_LOCAL_PATTERN = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def _load_vite_openai_key() -> str | None:
    """Read VITE_OPENAI_KEY from .env.local (the same file the browser dev server reads).
    Falls back to OPENAI_API_KEY then to no-key. .env.local lives at repo root; we read
    it directly rather than depending on dotenv (the venv doesn't ship it)."""
    env_local = REPO / ".env.local"
    if env_local.exists():
        try:
            with open(env_local) as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    m = _ENV_LOCAL_PATTERN.match(s)
                    if m and m.group(1) == "VITE_OPENAI_KEY":
                        return m.group(2).strip().strip('"').strip("'") or None
        except OSError:
            pass
    return os.environ.get("OPENAI_API_KEY") or None


# Schema kept in lock-step with src/lib/pipeline/vlmTriage.ts VLM_TRIAGE_SCHEMA.
# Changing one without the other is a contract drift — explicit so a future audit
# can grep for both names and confirm parity.
GPT_TRIAGE_SCHEMA = {
    "name": "submit_triage",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_quality": {"type": "string", "enum": ["diagnostic", "limited", "nondiagnostic"]},
            "projection": {"type": "string", "enum": ["pa_ap", "lateral", "unknown"]},
            "tb_screen_result": {
                "type": "string",
                "enum": ["screen_positive", "screen_negative", "abstain"],
            },
            "tb_score_uncalibrated": {"type": "number"},
            "confidence_band": {"type": "string", "enum": ["low", "medium", "high"]},
            "scar_shape_score_uncalibrated": {"type": "number"},
            "mimic_features_present": {"type": "array", "items": {"type": "string"}},
            "abnormality_localization": {"type": "array", "items": {"type": "string"}},
            "safety_flags": {"type": "array", "items": {"type": "string"}},
            "short_rationale": {"type": "string"},
            "refusal_or_limitation": {"type": ["string", "null"]},
            "model_version_seen_by_client": {"type": "string"},
        },
        "required": [
            "image_quality",
            "projection",
            "tb_screen_result",
            "tb_score_uncalibrated",
            "confidence_band",
            "scar_shape_score_uncalibrated",
            "mimic_features_present",
            "abnormality_localization",
            "safety_flags",
            "short_rationale",
            "refusal_or_limitation",
            "model_version_seen_by_client",
        ],
    },
}

GPT_PROMPT = (
    "You are a structured-output triage helper for a research preview tuberculosis screen. "
    "This is NOT diagnosis. Output ONLY the schema fields; do not narrate. "
    "When uncertain or when image quality is limited or nondiagnostic, choose ABSTAIN rather than screen_negative. "
    "Do not assume demographic priors. Do not infer clinical history from the image. "
    "Do not claim your scores are calibrated."
)


def _call_gpt(image_bytes: bytes, api_key: str, model: str = "gpt-5.5") -> dict:
    """Single deterministic call to gpt-5.5 via the Responses API, with the SAME
    structured-output schema the browser uses. Returns the parsed submission dict
    (already validated by the API's strict mode); raises on transport failure.

    We import openai lazily so the no-`--include-gpt` path doesn't depend on it."""
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": GPT_PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={"format": {"type": "json_schema", **GPT_TRIAGE_SCHEMA, "strict": True}},
    )
    # The Responses API exposes `.output_text` as the concatenated assistant text.
    raw = resp.output_text or "{}"
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _print_human(res: TriageResult, gpt_payload: dict | None) -> None:
    d = res.to_dict()
    verdict_color = {
        "tb": "\033[31m",  # red
        "no_tb": "\033[32m",  # green
        "abstain": "\033[33m",  # yellow
    }
    reset = "\033[0m"
    color = verdict_color.get(res.verdict, "")
    print(f"VERDICT  {color}{res.verdict.upper()}{reset}")
    print(f"tb_prob (calibrated under T={d['audit']['calibration']['T']:.4f})     "
          f"{d['tb_prob']:.4f}")
    print(f"tb_logit (raw, pre-calibration)                {d['tb_logit']:+.4f}")
    print(f"s_inactive (calibrated under T_seq={d['audit']['calibration']['T_sequelae']:.4f}) "
          f"{d['s_inactive']:.4f}")
    print(f"decided at threshold (the validated thr@95sens) {d['decided_at_threshold']:.4f}")
    if d["safety_net_applied"]:
        print(f"safety net applied:                            {d['safety_net_applied']}")
    if d["image_quality"]["warnings"]:
        print("image-quality warnings:")
        for w in d["image_quality"]["warnings"]:
            print(f"  - {w}")
    print("\nLATENCY  (ms; warm M4)")
    for k in ("harmonize", "seg", "rad_dino", "txrv", "heads", "total"):
        if k in d["latency_ms"]:
            print(f"  {k:<12s} {d['latency_ms'][k]:>6d}")
    print("\nAUDIT")
    print(f"  model_id     {d['audit']['model_id']}")
    print(f"  model_sha    {d['audit']['model_sha']}")
    print(f"  git_sha      {d['audit']['git_sha']}")
    print(f"  version      {d['audit']['version']}")
    print(f"  timestamp    {d['audit']['timestamp']}")
    if gpt_payload is not None:
        print("\n--- GPT-5.5 VISION SECOND OPINION (uncalibrated; consistency check) ---")
        agree_red = "\033[31m" if gpt_payload.get("tb_screen_result") != _local_to_screen(res.verdict) else "\033[32m"
        print(f"  tb_screen_result         {agree_red}{gpt_payload.get('tb_screen_result')}{reset}  "
              f"(local says {_local_to_screen(res.verdict)})")
        print(f"  tb_score_uncalibrated    {gpt_payload.get('tb_score_uncalibrated')}")
        print(f"  confidence_band          {gpt_payload.get('confidence_band')}")
        print(f"  short_rationale          {gpt_payload.get('short_rationale')!r}")


def _local_to_screen(v: str) -> str:
    """Map the local-pipeline verdict back to the VLM schema's screen_result enum for the
    agreement check. tb -> screen_positive, no_tb -> screen_negative, abstain -> abstain."""
    return {"tb": "screen_positive", "no_tb": "screen_negative", "abstain": "abstain"}[v]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Local-mode TB triage CLI (M22).")
    ap.add_argument("image", help="Path to a CXR image (jpg/png). DICOM not yet supported.")
    out_grp = ap.add_mutually_exclusive_group()
    out_grp.add_argument("--json", action="store_true", help="Print TriageResult as JSON (default: --human).")
    out_grp.add_argument("--human", action="store_true", help="Pretty single-screen output (default).")
    ap.add_argument(
        "--include-gpt",
        action="store_true",
        help="Also run gpt-5.5 vision via the OpenAI Responses API and report both numbers.",
    )
    ap.add_argument(
        "--gpt-model",
        default="gpt-5.5",
        help="OpenAI model id to use when --include-gpt is set (default: gpt-5.5).",
    )
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"error: image not found: {img_path}", file=sys.stderr)
        return 2
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    if args.include_gpt:
        api_key = _load_vite_openai_key()
        if not api_key:
            print(
                "error: --include-gpt set but no OpenAI key found. Set VITE_OPENAI_KEY in .env.local "
                "or OPENAI_API_KEY in the shell.",
                file=sys.stderr,
            )
            return 2
    else:
        api_key = None

    eng = TriageEngine()
    t0 = time.perf_counter()
    res = eng.run(image_bytes)
    load_plus_run_ms = int((time.perf_counter() - t0) * 1000)

    gpt_payload: dict | None = None
    if args.include_gpt and api_key:
        try:
            gpt_payload = _call_gpt(image_bytes, api_key, model=args.gpt_model)
        except Exception as e:  # transport / quota / refusal — surface, don't crash
            gpt_payload = {"error": repr(e)[:200]}

    if args.json:
        out = res.to_dict()
        out["wall_clock_run_ms"] = load_plus_run_ms
        if gpt_payload is not None:
            out["gpt_second_opinion"] = gpt_payload
        print(json.dumps(out, indent=2))
    else:
        _print_human(res, gpt_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
