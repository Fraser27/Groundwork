"""Shared constants.

Dependency-free on purpose so anything — including `src/graph/`, which nothing else
may import from — can import it without a cycle.
"""

from __future__ import annotations

PROJECT_NAME = "groundwork"

DEFAULT_AWS_REGION = "us-east-1"

#: Neptune's openCypher/Bolt port, and Neo4j's when the port is not remapped, so
#: one value covers both local dev and deployed.
GRAPH_PORT = 8182

#: AgentCore's MCP protocol contract fixes the container port at 8000, so the API
#: uses it too — one image, and local matches deployed.
APP_PORT = 8000

#: Ships as the default pack. The other three exist alongside it to keep the domain-agnostic
#: claim honest; if a second pack ever stops loading, the abstraction has rotted.
#:
#: Changing this changes which predicates are accepted at write time, and the packs are close to
#: disjoint: `legal` and `fintech` share only GOVERNED_BY, SUPERSEDES and MENTIONS. So a graph
#: holding facts written under one pack will have those writes rejected under another, and the
#: existing facts stay readable but no rule matches them. It is a per-tenant setting for that
#: reason -- this only decides where a tenant that has never chosen starts.
DEFAULT_ONTOLOGY_PACK = "retail"

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

#: Every default is Nova 2 Lite, so the system runs without access to Anthropic models. It is
#: reachable in accounts where Claude is not enabled, which is what a workshop needs, and it is the
#: cheapest thing on the list.
#:
#: The trade is real and worth stating: Nova is weaker at deciding what a passage *means*, which is
#: extraction's whole job. Expect more claims a reviewer has to correct, and more that read as
#: plausible rather than supported. Every model here is settable per tenant in Admin, and the
#: `method` string on each assertion records which model produced it, so raising extraction back to
#: Sonnet later does not orphan anything already extracted.
#:
#: Verified against the deployment account before switching: Nova 2 Lite returns clean JSON to a
#: JSON-only instruction, and accepts an image block, which OCR depends on.
DEFAULT_TEXT_MODEL = "global.amazon.nova-2-lite-v1:0"

#: Page transcription. A vision model rather than Textract because a legal document carries meaning
#: in charts, org charts, signature blocks and handwriting, which OCR returns nothing for. Reading
#: words off a page is mechanical, so a small model is a better fit here than anywhere else.
DEFAULT_OCR_MODEL = DEFAULT_TEXT_MODEL

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
DEFAULT_EXTRACTION_MODEL = DEFAULT_TEXT_MODEL
DEFAULT_SYNTHESIS_MODEL = DEFAULT_TEXT_MODEL

#: Text models an administrator may choose between, with what the trade-off is.
#:
#: A list rather than a live `ListInferenceProfiles` call: that returns everything the account can
#: reach, including vision-only and embedding models, and a dropdown offering a model that cannot
#: do the job is worse than a short list. Each id is a **global** inference profile verified
#: present in the deployment account -- a region-pinned id fails wherever the task is not, and
#: that failure reads as the model refusing the request rather than as a configuration mistake.
#:
#: `note` exists because "cheaper" is usually the reason to change this, and offering the choice
#: without saying what it costs in quality asks for a decision nobody can make.
#: Ordered default first, then by capability. The default leads because it is what a reader is
#: comparing against, and the notes say what moving away from it buys.
SELECTABLE_MODELS: tuple[tuple[str, str, str], ...] = (
    (
        DEFAULT_TEXT_MODEL,
        "Amazon Nova 2 Lite",
        (
            "The default everywhere, and the cheapest. Needs no access to Anthropic models. "
            "Weaker at judging what a passage means, so expect more extracted claims a reviewer "
            "has to correct, and watch the Retrieval transcript: a small model calls tools with "
            "wrong arguments more often."
        ),
    ),
    (
        "global.anthropic.claude-sonnet-5",
        "Claude Sonnet 5",
        "Most capable. Worth it for extraction, where deciding what a passage means is the job.",
    ),
    (
        "global.anthropic.claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "Close to Sonnet 5 and usually cheaper.",
    ),
    (
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "Claude Sonnet 4.5",
        (
            "A generation back and still strong at extraction. Picks the right entity kind far "
            "more reliably than the smaller models, which decides whether two documents naming "
            "one thing land on one node or two."
        ),
    ),
    (
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "Claude Haiku 4.5",
        (
            "Between Nova and Sonnet. Good for transcription and straightforward extraction, "
            "weaker at judging what a passage means."
        ),
    ),
)

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
