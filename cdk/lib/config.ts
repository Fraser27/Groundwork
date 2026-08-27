import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';

export const PROJECT = 'Groundwork';

/**
 * Lowercase form of `PROJECT`, for the resource names that reject mixed case.
 *
 * Cognito's domain prefix and every OpenSearch Serverless name (collection, collection
 * group, and each security/access policy) are validated against a lowercase-only
 * pattern -- `^[a-z][a-z0-9-]{2,31}$` for OpenSearch, `^[a-z0-9-]+$` for the Cognito
 * prefix. `PROJECT` itself stays PascalCase because it is also the stack-id prefix, a
 * Cedar entity type name, and CloudWatch log stream prefixes -- none of which enforce
 * this, and PascalCase reads better there. Use this constant instead of lower-casing
 * `PROJECT` ad hoc at each call site, so the two never drift.
 */
export const PROJECT_SLUG = PROJECT.toLowerCase();

/*
 * Two feature flags in cdk.json are load-bearing, and JSON cannot hold a comment
 * explaining why, so they are documented here:
 *
 * `@aws-cdk/core:validateAgainstDefaultRules` — promotes CloudFormation schema
 * warnings to synth errors. Worth it because the alternative is discovering an
 * invalid property after 15 minutes of Neptune creation.
 *
 * `@aws-cdk/core:defaultCrossStackReferences: strong` — `data` refuses to remove
 * an export that `app` still consumes, so a careless change to the graph tier
 * cannot silently break the API. The cost is that renaming a shared resource
 * takes two deploys, which is the right trade for the tier holding the only
 * irreplaceable state.
 */

/**
 * Every knob that we expect to turn without editing a stack.
 *
 * These live in `cdk.json` context rather than as CloudFormation parameters
 * because CDK resolves context at synth time, so `cdk diff` shows the real
 * consequence of a change. A CFN parameter would hide it until deploy.
 */
export interface GroundworkConfig {
  /** Raise this off the burstable class before any load test — see cdk/README.md. */
  readonly neptuneInstanceClass: string;
  readonly neptuneInstanceCount: number;
  /** Must track the engine's major line: `neptune1.4` for 1.4.x, `neptune1.3` for 1.3.x. */
  readonly neptuneParameterGroupFamily: string;
  readonly vectorMinOcu: number;
  readonly vectorMaxOcu: number;
  readonly appDesiredCount: number;
  readonly appCpu: number;
  readonly appMemoryMiB: number;
  readonly defaultOntology: string;
  /**
   * The one tenant whose platform-admins may create and delete other tenants.
   *
   * Not "any platform-admin": that is a role within a firm, so it would let one customer
   * delete another. Empty closes those routes entirely, which is the safe reading of an
   * operator tenant nobody configured.
   */
  readonly homeTenant: string;
  /**
   * Availability Zone *names* for the VPC, e.g. `['us-east-1a','us-east-1b']`.
   *
   * Left unset by default because AZ names are shuffled per account: the
   * `us-east-1a` in your account is a different physical zone from mine. The
   * constraint we actually need is on AZ *IDs* (see network-stack.ts), so this
   * has to be resolved per account before the first deploy.
   */
  readonly availabilityZones?: string[];
  /**
   * Deploy the AgentCore MCP runtime, and with it build the image for ARM64.
   *
   * One flag for both because AgentCore Runtime accepts ARM64 only, and it runs `app`'s
   * image verbatim. So there is no configuration where the runtime exists and the image
   * is x86_64. Off by default: on an x86_64 build host the ARM64 build needs QEMU, which
   * turns `pip install` into a multi-minute emulated one, and the MCP tools are still
   * served in-process to the Retrieval agent either way (see MCP_PORT). Turning this on
   * costs a cross-build; leaving it off costs only the third-party MCP endpoint.
   */
  readonly agentCoreMcp: boolean;
}

export function readConfig(scope: Construct): GroundworkConfig {
  const ctx = <T>(key: string, fallback: T): T => {
    const v = scope.node.tryGetContext(key);
    return v === undefined || v === null ? fallback : (v as T);
  };

  const azs = ctx<string[] | undefined>('availabilityZones', undefined);
  if (azs && azs.length < 2) {
    throw new Error(
      'availabilityZones must list at least 2 zones: Neptune requires a subnet ' +
        'group spanning multiple AZs, and a single-AZ VPC endpoint has no redundancy.',
    );
  }

  return {
    neptuneInstanceClass: ctx('neptuneInstanceClass', 'db.t4g.medium'),
    neptuneInstanceCount: ctx('neptuneInstanceCount', 1),
    neptuneParameterGroupFamily: ctx('neptuneParameterGroupFamily', 'neptune1.4'),
    vectorMinOcu: ctx('vectorMinOcu', 0),
    vectorMaxOcu: ctx('vectorMaxOcu', 4),
    appDesiredCount: ctx('appDesiredCount', 1),
    appCpu: ctx('appCpu', 512),
    appMemoryMiB: ctx('appMemoryMiB', 1024),
    // Kept in step with `src/constants.DEFAULT_ONTOLOGY_PACK`. This value is passed as
    // ONTOLOGY_PACK and so wins over the Python default, which means a mismatch here silently
    // deploys a different vocabulary than the code says it ships.
    defaultOntology: ctx('defaultOntology', 'fintech'),
    homeTenant: ctx('homeTenant', 'demo-firm'),
    availabilityZones: azs,
    agentCoreMcp: ctx('agentCoreMcp', false),
  };
}

/**
 * Where a presigned upload lands, before it has been hashed.
 *
 * Must match `RAW_PREFIX` in src/documents/storage.py: the notification filter is
 * declared here and the key is parsed there, so a change to one and not the other
 * yields a bucket whose uploads silently never trigger ingestion.
 */
export const RAW_PREFIX = 'raw/';

/**
 * Ingest throughput knobs, mirroring the defaults in `src/constants.py`.
 *
 * Set explicitly in the task environment rather than left to the image's defaults so
 * that `cdk diff` shows a change to them — a silent default is how a Bedrock throttling
 * incident becomes hard to explain.
 */
export const PAGE_BATCH_SIZE = 5;
export const PAGE_CONCURRENCY = 8;
export const MAX_CONCURRENT_INGESTS = 4;

/** Port openCypher-over-Bolt listens on. Neptune's default; not configurable here. */
export const NEPTUNE_PORT = 8182;

/**
 * AgentCore's MCP protocol contract fixes the container port at 8000, so the API
 * uses it too — one Dockerfile, and local dev matches deployed.
 */
export const APP_PORT = 8000;

/**
 * Where the MCP sidecar listens, on the task's loopback only.
 *
 * Not published by the load balancer. This port is what the Retrieval agent in the API
 * container calls, so the tools work whether or not `agentCoreMcp` is set; that flag adds
 * the *third-party* entry point, an AgentCore runtime that verifies a Cognito token first.
 */
export const MCP_PORT = 8001;

export function tagStack(stack: cdk.Stack, component: string): void {
  cdk.Tags.of(stack).add('Project', PROJECT);
  cdk.Tags.of(stack).add('Component', component);
  assertAsciiDescriptions(stack);
}

/**
 * Fail synth if any resource description contains a non-ASCII character.
 *
 * Several AWS services reject them as "non-printable control characters" — Neptune,
 * RDS, EC2 security groups and IAM among them. An em dash is the usual culprit, and this
 * whole repo writes prose with em dashes, so the mistake is a natural one to make.
 *
 * It exists because the failure is expensive and misleading: Neptune accepted the
 * template, then failed 5 minutes into a 25-minute deploy and rolled the entire data
 * tier back. The error names a "control character" in a string that visibly contains
 * none. Catching it at synth turns that into a one-line local failure.
 */
function assertAsciiDescriptions(stack: cdk.Stack): void {
  const offenders: string[] = [];

  cdk.Aspects.of(stack).add({
    visit(node) {
      const resource = node as unknown as { cfnResourceType?: string };
      if (!resource.cfnResourceType) return;
      const props = (node as unknown as { _cfnProperties?: Record<string, unknown> })
        ._cfnProperties;
      if (!props) return;

      for (const [key, value] of Object.entries(props)) {
        if (!/escription$/i.test(key) || typeof value !== 'string') continue;
        // eslint-disable-next-line no-control-regex
        if (/[^\x00-\x7F]/.test(value)) {
          offenders.push(`${node.node.path} .${key}: ${JSON.stringify(value)}`);
        }
      }
    },
  });

  // Aspects run at synth, after the tree is built, so the check is registered here and
  // reported by this validation hook rather than inline.
  stack.node.addValidation({
    validate: () =>
      offenders.length === 0
        ? []
        : [
            'Non-ASCII character in a resource description. Neptune, RDS, EC2 and IAM ' +
              'reject these as control characters, failing mid-deploy. Use "-" not an em dash:\n  ' +
              offenders.join('\n  '),
          ],
  });
}
