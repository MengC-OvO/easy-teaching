# Local Privacy & Safety Gateway

EasyTeaching uses a separate local process for privacy detection, prompt-safety
classification, deterministic redaction, and policy signals. Both processes can
live in this monorepo while retaining different dependency and failure boundaries.

> **Current status:** experimental and disabled by default. The 240-case live
> Gateway evaluation failed its release gate (`92.5%` HTTP success, PII recall
> `60%`, PII F1 `75%`, phone recall `0%`). Fail-closed behavior prevented
> plaintext response leakage, but the adapter must be retrained on premasked
> production-format inputs before real-data use.

```text
Browser -> EasyTeaching API :8000 -> Local Safety Gateway :8010
                                      |
                                      +-> rules + local Qwen adapter
```

## Runtime pipeline

The gateway runs this sequence without sending source text to an external service:

1. high-confidence regex rules pre-mask email, phone, and contextual DOB values;
2. the local Qwen v11 LoRA adapter emits semantic labels and remaining entities;
3. Pydantic rejects malformed or invented model values;
4. deterministic Python policy derives `allow`, `clarify`, or `block`;
5. deterministic Python constructs final placeholders;
6. plaintext mappings enter a one-process TTL vault and HTTP receives only an opaque id;
7. EasyTeaching restores the final draft content once through `/v1/restore`;
8. the redacted GraphState/checkpoint remains unchanged while the authorized
   API-facing draft snapshot stores the restored content for repeat reads.

The process remains fail-closed:

- `GET /health` returns 200 when the process is alive.
- `GET /ready` returns 200 only when the model and adapter are loaded.
- `GET /ready` returns 503 otherwise.
- `POST /v1/inspect` returns 503 when local inspection cannot complete safely.

The EasyTeaching message route now inspects before creating a conversation run,
writing a graph checkpoint, or calling ReAct. Only the redacted text and opaque
mapping id enter GraphState in `enforce` mode. Real personal or production data
remains out of scope for this synthetic-only project.

## Contract

A request contains a caller-generated UUID, an optional session UUID, a bounded
text value, and its source kind. The response contains semantic safety signals,
a deterministic action, redacted text for allowed requests, an opaque mapping id,
and entity counts. It never contains plaintext mapping values.

ReAct, RAG, checkpoints, external model providers, and ordinary logs must never
receive the mapping vault contents. `/v1/restore` is local-only, consumes a mapping
once, and is intended solely for the final response boundary.

## EasyTeaching input modes

- `disabled`: no gateway call; this remains the default during integration.
- `shadow`: inspect and immediately discard mappings, but forward the original;
  configuration rejects this mode outside local/test/development environments.
- `enforce`: `allow` forwards only redacted text, `clarify` returns HTTP 422,
  `block` returns HTTP 403, and gateway failure returns HTTP 503 before run creation.

An enabled gateway URL must resolve to loopback (`127.0.0.1`, `localhost`, or
`::1`). In `enforce` mode the opaque mapping id is checkpoint-safe, Agent and
GraphState see only placeholders, and deterministic restoration happens before
the completed draft snapshot is published. Failed/no-draft runs discard their
mapping. A later turn explicitly clears the previous turn's mapping id.

`conversation_run_results` is therefore an authorized application data boundary:
in real deployment its restored draft may contain personal information and needs
database encryption, access control, retention, and deletion policy. The current
gateway vault is still in-memory; replacing it with encrypted durable storage is
the next resilience step so a gateway restart cannot lose an in-flight mapping.

## Local model assets

Model weights and adapters belong under the ignored `local_models/` directory or
at paths supplied by `SAFETY_MODEL_DIR` and `SAFETY_ADAPTER_DIR`. They are never
committed. `SAFETY_MODEL_BACKEND=auto` selects CUDA first and Apple MPS second;
it never silently falls back to CPU. Use `requirements-model-cuda.txt` on an
NVIDIA machine and `requirements-model-mac.txt` on an Apple Silicon Mac.

The CUDA backend uses bitsandbytes NF4 4-bit weights with BF16/FP16 computation.
The MPS backend uses FP16 because bitsandbytes CUDA kernels do not run on Apple
GPU hardware. Both use the same frozen Qwen base model, PEFT LoRA adapter, strict
JSON parsing, HTTP contract, and deterministic redaction pipeline.

## Windows setup and verification

Create the independent runtime without copying model weights:

```powershell
.\scripts\setup_safety_gateway.ps1 `
  -ModelDir "C:\path\to\Qwen2.5-1.5B-Instruct" `
  -AdapterDir "C:\path\to\qlora-formal-v11\best-adapter"
```

Start the single-worker GPU service:

```powershell
.\scripts\start_safety_gateway.ps1
```

Run the synthetic end-to-end verification. It starts the gateway itself
when port 8010 is not already serving a ready instance:

```powershell
.\scripts\demo_privacy_flow.ps1
```

## Apple Silicon Mac setup and verification

Use Python 3.10 or newer. A 16 GB Mac is recommended for Qwen2.5-1.5B FP16
inference. This creates an independent environment without copying or uploading
model assets:

```bash
bash ./scripts/setup_safety_gateway.sh \
  /path/to/Qwen2.5-1.5B-Instruct \
  /path/to/qlora-formal-v11/best-adapter
```

Start the gateway or run the complete synthetic verification:

```bash
bash ./scripts/start_safety_gateway.sh
bash ./scripts/demo_privacy_flow.sh
```

The MPS path needs a one-time real-device smoke test after checkout. Windows can
verify selection and shared contracts, but cannot execute Apple's Metal kernels.

## Final synthetic Gateway evaluation

With the real local Qwen2.5-1.5B v11 adapter, deterministic premasking, HTTP
Gateway, mapping vault and one-time restoration, the independent challenge suite
produced:

| Metric | Result |
| --- | ---: |
| Cases | `240` |
| HTTP success | `92.5%` |
| Injection 3-class accuracy, all cases | `86.25%` |
| Block precision / recall / F1 | `100% / 100% / 100%` |
| PII precision / recall / F1 | `100% / 60% / 75%` |
| Exact restore / one-time consumption | `32/50 / 32/50` |
| Plaintext leakage in Gateway responses | `0%` |
| Latency P50 / P95 | `1.78 s / 3.86 s` |
| Release gate | **Failed** |

The confirmed cause is a training/serving distribution mismatch: rules replace
some values with `<PREMASKED_...>` before inference, while the adapter was mainly
evaluated on raw text and sometimes returns a reserved token or a value absent
from its input. Strict validation then correctly fails closed with HTTP 503.
Do not loosen that validation to improve the headline success rate; retrain and
re-run the frozen production-pipeline suite.
