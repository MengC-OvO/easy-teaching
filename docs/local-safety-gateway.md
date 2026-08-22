# Local Privacy & Safety Gateway

EasyTeaching uses a separate local process for privacy detection, prompt-safety
classification, deterministic redaction, and policy signals. Both processes can
live in this monorepo while retaining different dependency and failure boundaries.

```text
Browser -> EasyTeaching API :8000 -> Local Safety Gateway :8010
                                      |
                                      +-> rules + local Qwen adapter (next step)
```

## Step-one boundary

This first implementation defines HTTP contracts and liveness/readiness behavior.
It deliberately does not load Qwen and cannot inspect user messages yet. The
default gateway is therefore fail-closed:

- `GET /health` returns 200 when the process is alive.
- `GET /ready` returns 503 until the real pipeline is loaded.
- `POST /v1/inspect` returns 503 until the real pipeline is loaded.

The EasyTeaching message route is not connected in this step. Merely starting
the skeleton cannot cause raw text to bypass the existing application behavior.

## Contract

A request contains a caller-generated UUID, an optional session UUID, a bounded
text value, and its source kind. The response will eventually contain:

- semantic safety signals;
- a deterministic `allow`, `clarify`, or `block` action;
- redacted text for allowed requests;
- an opaque mapping identifier, never the raw mapping;
- entity counts by label, never entity values.

The HTTP response intentionally cannot carry plaintext mapping values. ReAct,
RAG, checkpoints, external model providers, and ordinary logs must never receive
the mapping vault contents.

## Modes reserved in EasyTeaching settings

- `disabled`: no gateway call; local development only.
- `shadow`: synthetic/local diagnostics only; never use it before an external LLM.
- `enforce`: gateway decisions and redaction are mandatory.

Step one leaves the default at `disabled`. A later integration step will reject
unsafe mode/environment combinations and place the call before persistence.
