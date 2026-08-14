"""Shared constants.

Dependency-free on purpose so anything — including `src/graph/`, which nothing else
may import from — can import it without a cycle.
"""

from __future__ import annotations

PROJECT_NAME = "lexgraph"

DEFAULT_AWS_REGION = "us-east-1"

#: Neptune's openCypher/Bolt port, and Neo4j's when the port is not remapped, so
#: one value covers both local dev and deployed.
GRAPH_PORT = 8182

#: AgentCore's MCP protocol contract fixes the container port at 8000, so the API
#: uses it too — one image, and local matches deployed.
APP_PORT = 8000

#: Ships as the default pack. `healthcare` exists to keep the domain-agnostic
#: claim honest; if a second pack ever stops loading, the abstraction has rotted.
DEFAULT_ONTOLOGY_PACK = "legal"

#: Trust floor for retrieval, mirroring `src.graph.scope.DEFAULT_MIN_CONFIDENCE`.
#: Duplicated rather than imported because this module must not depend on
#: `src.graph`, and a config default belongs where config is read.
DEFAULT_MIN_CONFIDENCE = 0.8

#: Chunking. Offsets are stored on every assertion's source locator, so changing
#: these does NOT re-slice existing chunks — it only affects documents ingested
#: afterwards. Old assertions keep citing offsets into the text they were read from.
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_CHUNK_OVERLAP_CHARS = 200

#: Titan v2's native width. Changing this invalidates the whole vector index,
#: because a kNN search across two dimensionalities is not an error — it is
#: silently meaningless.
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMENSIONS = 1024

#: Page transcription. A cheaper, faster model than extraction on purpose: reading
#: words off a page is mechanical, whereas deciding what they mean is not. A vision
#: model is used rather than Textract because a legal document carries meaning in
#: charts, org charts, signature blocks and handwriting, which OCR returns nothing for.
DEFAULT_OCR_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

#: Pages per transcription batch, and how many vision calls may be in flight for one
#: document. Batching is the unit of progress reporting and of a confined failure;
#: concurrency is the throughput knob. One call per page means 400 sequential pages
#: take tens of minutes, while 400 at once earns throttling and a burst of retries.
PAGE_BATCH_SIZE = 5
MAX_PAGE_CONCURRENCY = 8

#: Documents ingesting at once in one process. Multiplies with MAX_PAGE_CONCURRENCY, so
#: this is what stops a bulk upload becoming 4x8 in-flight Bedrock calls.
MAX_CONCURRENT_INGESTS = 4

#: Extraction models. The versioned `method` string on each assertion records
#: which one produced it, so these can change without orphaning past extractions.
DEFAULT_EXTRACTION_MODEL = "global.anthropic.claude-sonnet-5"
DEFAULT_SYNTHESIS_MODEL = "global.anthropic.claude-sonnet-5"

#: Cap on rows returned from a structured query. A governed metric that would
#: return more is answered with an aggregate, not a truncated table.
MAX_QUERY_ROWS = 500

#: Cedar action names. Kept here so a typo is one shared constant rather than a
#: string literal in each call site — and a typo'd action in Cedar denies
#: silently, which looks exactly like a correct denial.
ACTION_READ_MATTER = "ReadMatter"
ACTION_WRITE_ASSERTION = "WriteAssertion"
ACTION_REVIEW_ASSERTION = "ReviewAssertion"
ACTION_RUN_QUERY = "RunQuery"
