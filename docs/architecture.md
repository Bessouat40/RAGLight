# Architecture

RAGLight 4 has a dependency-free core and explicit optional integrations.

```text
                         +----------------+
                         | contracts/models|
                         | registry core  |
                         +-------^--------+
                                 |
                 +---------------+---------------+
                 |                               |
        +--------+---------+            +--------+---------+
        |     pipeline     |            |   integrations   |
        |  orchestration   |            | provider adapters|
        +------------------+            +------------------+
```

Dependencies point toward contracts and models. The pipeline and integrations
are siblings: the pipeline receives components through protocols and never
imports a concrete provider.

## Package ownership

| Package | Responsibility |
| --- | --- |
| `raglight.contracts` | Stable structural protocols for extensions |
| `raglight.models` | Provider- and transport-independent values |
| `raglight.registry` | Provider-independent lazy registry primitives |
| `raglight.pipeline` | Classic retrieval, optional reranking, and generation |
| `raglight.integrations` | Optional provider adapters and SDK translation |

`raglight.RAGPipeline` and `raglight.PipelineConfig` are the root facade.
Internal implementation modules are not public merely because Python can
import them.

## Core rules

- contracts, models, and registry primitives do not import integrations or
  optional SDKs.
- `pipeline` imports core registry types, contracts, and models, never
  integrations.
- importing `raglight` does not load any optional provider package.
- domain data uses `raglight.models.Document`, not an SDK document type.
- configuration for credentials and provider clients stays at the integration
  boundary.
- conversation history is supplied by the caller and is never global state.

These rules are executable in `scripts/check_architecture.py` and
`tests/architecture/`.

## Classic versus agentic RAG

The current v4 package contains only the classic deterministic flow:

```text
question -> retrieve -> optional rerank -> generate -> answer
```

Agentic execution will be implemented as a separate capability with its own
public namespace, dependencies, tools, step budget, and termination behavior.
It will reuse stable contracts where their semantics match, but will not
inherit from `RAGPipeline` and will not add an `agentic` mode flag to it.

The v3 agentic runtime, API server, CLI, and UI were removed during the v4
break. Any future versions of those capabilities must be rebuilt on these v4
boundaries rather than restored as compatibility code.

## Integration rules

Each integration owns:

- external SDK imports and actionable missing-dependency errors;
- provider-specific authentication and configuration;
- translation between SDK types and core models;
- deterministic provider unit tests and shared contract coverage.

Adding an integration must not add provider-name branches to the pipeline.
Built-in names are registered as lazy import references in
`raglight.integrations`; resolving one entry imports only its provider module.

## Cross-cutting invariants

- Tests are deterministic and offline by default.
- Errors identify the responsible capability and installation extra.
- Secrets never appear in logs, exceptions, fixtures, or snapshots.
- Public behavior is tested through public imports.
- Major architectural decisions are recorded in `docs/decisions/`.
