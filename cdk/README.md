# Groundwork infrastructure

An operator's reference: what each stack holds, what it costs, what to tune, and what
survives a teardown.

**Installing is not here.** `../README.md` covers a fresh account, and
`../scripts/deploy.sh` does it in one non-interactive command, including the two things
this file used to document as manual steps: resolving availability zones, and the
two-pass `webOrigin` that closes the circular callback requirement. Duplicating them here
is how this file came to describe a second pass that redeployed only `GroundworkAuth`,
leaving the S3 CORS rule unset and browser uploads failing.

| Stack | Holds | Redeployed |
|---|---|---|
| `GroundworkNetwork` | VPC, subnets, security groups, VPC endpoints | rarely |
| `GroundworkData` | Neptune, OpenSearch Serverless, DynamoDB, S3 | rarely |
| `GroundworkAuth` | Cognito user pool + hosted UI, Cedar policy store | rarely |
| `GroundworkApp` | ECS Fargate (FastAPI) behind an ALB | constantly |
| `GroundworkMcp` | MCP server on Bedrock AgentCore Runtime | often, and only if `agentCoreMcp` |
| `GroundworkWeb` | CloudFront + S3 for the React UI | often |

`GroundworkMcp` is opt-in via the `agentCoreMcp` context flag, off by default, because
AgentCore Runtime accepts ARM64 images only and it runs `app`'s image verbatim. With the
flag off the image and the Fargate task are built for the build host's architecture, so
an x86_64 machine needs no QEMU; with it on, a cross-build is registered by `deploy.sh`.
Either way the MCP tools run as a sidecar on the API task, which is what the Retrieval
agent calls. What the flag buys is the authenticated endpoint an *outside* MCP client
connects to.

Turning the flag off does not remove an already-deployed `GroundworkMcp`: CDK stops
managing the stack rather than deleting it, so the runtime keeps running against a stale
ARM64 image. Run `npx cdk destroy GroundworkMcp` before flipping it off.

The split is by **deploy cadence and blast radius**, not by feature. `data` is separate
because Neptune takes about 15 minutes to create and holds the only state that cannot be
rebuilt; `app` is separate because it is redeployed several times a day and a rollback
there must not take a CloudFormation lock on the graph.

## Everyday commands

```bash
npx cdk synth --quiet    # every stack, no AWS calls
npx cdk diff             # what a deploy would change
npx cdk deploy --all     # 25-30 min from cold, most of it Neptune
```

`synth` and `diff` operate on every stack by default and reject `--all`; `deploy` and
`destroy` require it. `make synth` and `make deploy` wrap both correctly.

## Availability zones

`SUPPORTED_AZ_IDS` in `lib/network-stack.ts` is the intersection of the zones
**AgentCore Runtime** supports for VPC connectivity and those the **OpenSearch
Serverless** data-plane endpoint is offered in. The intersection is smaller than either
list, and a subnet in the wrong zone fails `GroundworkMcp` with an error naming the
*subnet* rather than the zone. Kept as the intersection even with `agentCoreMcp` off, so
that turning the flag on later does not need the VPC rebuilt.

`deploy.sh` resolves this per account. Add a region to that map before deploying
somewhere new: those are zone **IDs**, and AZ *names* are shuffled per account, so
`us-east-1a` is a different physical zone in every account.

## Tunable context

All in `cdk.json`, read by `lib/config.ts`. Context rather than CloudFormation
parameters so `cdk diff` shows the real consequence of a change instead of hiding
it until deploy.

| Key | Default | Notes |
|---|---|---|
| `neptuneInstanceClass` | `db.t4g.medium` | Raise before any load test — see below |
| `neptuneInstanceCount` | `1` | Writer only; >1 adds read replicas |
| `neptuneParameterGroupFamily` | `neptune1.4` | Must track the engine's major line |
| `vectorMinOcu` | `0` | 0 enables NextGen scale-to-zero |
| `vectorMaxOcu` | `4` | Spend ceiling |
| `appCpu` / `appMemoryMiB` | `512` / `1024` | Fargate task size |
| `appDesiredCount` | `1` | No redundancy at 1 |
| `defaultOntology` | `retail` | Pack from `ontologies/`, must match `constants.DEFAULT_ONTOLOGY_PACK` |
| `availabilityZones` | unset | See above |
| `webOrigin` | unset | Set after the first deploy |

## Cost

Rough `us-east-1` monthly figures, one instance, light use. **Be sceptical of the
total** — it assumes an idle vector store and no meaningful document volume. Page
transcription is one Bedrock vision call per page, so a large scanned bundle is a
real, usage-driven cost that none of these lines predicts.

| Item | Cost | Scales with |
|---|---|---|
| Neptune `db.t4g.medium` | **free for 750 hrs, then ~$50/mo** | wall-clock time, always |
| Neptune storage + I/O | ~$1–10 | graph size, query volume |
| NAT gateway | ~$33 + data | always on |
| ALB | ~$17 + LCUs | always on |
| Fargate (0.5 vCPU/1 GB) | ~$18 | task count × uptime |
| Interface VPC endpoints (8 × 2 AZs) | ~$117 | AZ count × endpoint count |
| OpenSearch Serverless | **$0 idle**, ~$175/OCU-month active | traffic |
| DynamoDB, S3, CloudFront | a few dollars | usage |
| AgentCore Runtime | per-invocation | tool calls |
| **Idle floor** | **~$195–250/mo** | |

Two things to be honest about:

**Neptune Database has no scale-to-zero.** It bills continuously from creation to
deletion whether or not a single query is run. After the 750 free instance-hours
(~31 days of one instance) that is ~$50/mo forever. This is the single biggest
reason not to leave a dev stack up over a holiday. If you need a graph that costs
nothing while idle, Neptune Database is the wrong product — but Neptune Analytics
has a 32 m-NCU floor per graph and no autoscaling, which suits multi-tenant SaaS
even less.

**The eight interface endpoints cost more than the Fargate service.** At ~$7.20/mo
per endpoint per AZ, two AZs makes ~$58/mo. They buy the ability to run the app in
private subnets without paying NAT data-processing on every Bedrock call, and they
are the right shape for handling privileged documents. For a throwaway dev stack
they are the first thing to cut.

### `db.t4g.medium` is not a production instance

Free-tier eligible for 750 hours, Graviton so cheaper than `t3`, and AWS says
plainly it is not for production. Concretely, on the `T` family:

- 2:1 RAM-to-vCPU (vs 8:1 on `R`) disables the DFE engine statistics that make
  **openCypher** fast — and openCypher is the only query language we use
- large traversals raise `OutOfMemoryException`
- Graph Explorer and the Neptune GenAI integrations do not work
- AWS explicitly advises against load testing on it; results are not indicative

This is intended for now. Raise `neptuneInstanceClass` to `db.r6g.large` or larger
before any performance work — it is a context change, not a code edit. Watch
`BufferCacheHitRatio` (below 99.9% means not enough RAM) and
`MainRequestQueuePendingRequests` (above zero means not enough vCPU).

## Teardown gotchas

```bash
npx cdk destroy --all
```

**Resources that survive by design.** Each is deliberate; each keeps billing.

| Resource | Policy | Why |
|---|---|---|
| Document bucket | `RETAIN` | S3 is the only source of truth. Neptune and the vector index are derived — this is not. |
| Document KMS key | `RETAIN` | Destroying it makes every object in the retained bucket permanently unreadable. |
| `TenantTable` | `RETAIN` | Maps a JWT claim to a live tenant. Losing it orphans every S3 prefix and graph node. |
| `GrantTable` | `RETAIN` | Ethical walls. A lost denylist entry is a privilege breach, not an outage. |
| Neptune cluster | `SNAPSHOT` | A final snapshot is taken and kept. It bills as storage. |
| Cognito user pool | `DESTROY` | Currently destroyed. **Change to `RETAIN` before the first real tenant** — there is no import path back. |

**The document bucket is versioned and RETAINed**, so emptying it means deleting
every version, not every key. Object Lock is off (see below), so no bypass header is
needed.

**Other things that block or linger:**

- **Deletion order.** Destroy stacks in reverse (`web`, `mcp`, `app`, `auth`,
  `data`, `network`), or the strong cross-stack exports refuse. `cdk destroy --all`
  handles this; destroying `GroundworkData` alone will not.
- **The VPC will not delete** while the OpenSearch Serverless VPC endpoint's ENIs
  exist. It resolves on its own after a few minutes; retry rather than hand-deleting.
- **The Cognito domain prefix** is `groundwork-<account-id>` and is globally unique.
  Recreating too soon after a destroy can collide while the old one drains.
- **ECR images** from `DockerImageAsset` live in the CDK bootstrap asset repository
  and are not deleted with the stack. They accumulate; each is a few hundred MB.
- **CloudWatch log groups** for Neptune audit logs and the AgentCore runtime persist
  independently.
- **The final Neptune snapshot** is charged as snapshot storage indefinitely. Delete
  it explicitly when you are certain.

To tear down and genuinely stop all charges, after `cdk destroy --all`: delete the
final Neptune snapshot, empty and delete the document bucket (all versions),
schedule the KMS key for deletion, drop the two retained tables, and prune the ECR
repository.

## Object Lock: enable it before production

The document bucket ships **without** S3 Object Lock. For a system of record holding
privileged material that is the wrong long-term default, so this is a deliberate
staging decision rather than a recommendation.

Why it is off now:

- **Object Lock can only be enabled when a bucket is created.** There is no way to
  add it later, and no way to remove it once set. Getting it wrong in either
  direction costs a bucket migration.
- Uploads arrive by **presigned POST under `raw/`**, so a default retention would
  lock every abandoned, mistaken or malicious upload for the full period. Clearing a
  typo would require `s3:BypassGovernanceRetention`.
- The processed key is **content-addressed**, so the object that lands is not yet the
  object that becomes the record. Locking on arrival locks the wrong thing.

For production, create the bucket with `objectLockEnabled: true` and apply retention
**per object on the processed copy**, once the bytes are hashed and the document has
actually entered the record — not as a bucket-wide default:

```ts
objectLockEnabled: true,
// No objectLockDefaultRetention: retention is set per object after hashing.
```

then pass `ObjectLockMode: 'GOVERNANCE'` and an `ObjectLockRetainUntilDate` on the
`copy_object` call that writes the processed key, and keep the `raw/` prefix
unlocked so the lifecycle rule can still expire it.

Use `GOVERNANCE`, not `COMPLIANCE`. Under `COMPLIANCE` nobody — including AWS root —
can delete before the retention period expires, which would make a GDPR erasure
request impossible to honour. `GOVERNANCE` keeps erasure possible for an operator
holding the bypass permission, which is the point.
