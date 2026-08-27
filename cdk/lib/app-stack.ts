import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as neptune from 'aws-cdk-lib/aws-neptune';
import * as aoss from 'aws-cdk-lib/aws-opensearchserverless';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import * as path from 'path';

import {
  APP_PORT,
  MCP_PORT,
  GroundworkConfig,
  MAX_CONCURRENT_INGESTS,
  NEPTUNE_PORT,
  PAGE_BATCH_SIZE,
  PAGE_CONCURRENCY,
  PROJECT,
  PROJECT_SLUG,
  RAW_PREFIX,
  tagStack,
} from './config';

export interface AppStackProps extends cdk.StackProps {
  readonly config: GroundworkConfig;
  readonly vpc: ec2.IVpc;
  readonly appSg: ec2.ISecurityGroup;
  readonly albSg: ec2.ISecurityGroup;
  readonly neptuneCluster: neptune.CfnDBCluster;
  readonly neptuneEndpoint: string;
  readonly vectorCollection: aoss.CfnCollection;
  readonly vectorCollectionEndpoint: string;
  readonly documentBucket: s3.IBucket;
  readonly athenaResultsBucket: s3.IBucket;
  readonly documentKey: kms.IKey;
  readonly tenantTable: dynamodb.ITable;
  readonly jobTable: dynamodb.ITable;
  readonly grantTable: dynamodb.ITable;
  readonly userPoolId: string;
  readonly userPoolClientId: string;
  readonly issuerUrl: string;
  readonly policyStoreId: string;
}

/**
 * FastAPI on Fargate behind an ALB — control plane, serve path, and the ingestion
 * workers, in one service.
 *
 * Redeployed constantly, which is why it is its own stack: a rollback here should
 * not put a CloudFormation lock on Neptune. It also owns the ECR image asset, so
 * `cdk deploy` of `app` alone is the whole application deploy loop.
 *
 * Workers share the image and the task role rather than running as a separate
 * service. They need the same graph client, the same scope module and the same
 * ontology, and splitting them would be two deploy artefacts that must stay in
 * lockstep. Split them out when ingestion load and query load actually diverge.
 */
export class AppStack extends cdk.Stack {
  readonly loadBalancer: elbv2.ApplicationLoadBalancer;
  readonly taskRole: iam.Role;
  /** The image the MCP stack also runs, so tools and API can never drift apart. */
  readonly image: ecrAssets.DockerImageAsset;
  /** Shared with `mcp`, which overrides only APP_MODULE. */
  readonly containerEnvironment: Record<string, string>;

  constructor(scope: Construct, id: string, props: AppStackProps) {
    super(scope, id, props);
    tagStack(this, 'app');

    const { config, vpc, appSg, albSg } = props;

    this.image = new ecrAssets.DockerImageAsset(this, 'AppImage', {
      directory: path.join(__dirname, '../..'),
      // ARM64 because AgentCore Runtime accepts nothing else, and the MCP stack
      // runs this same image. Fargate is happy either way, so matching costs us
      // nothing and removes a second build.
      platform: ecrAssets.Platform.LINUX_ARM64,
      // This list is NOT redundant with .dockerignore — the two run at different
      // stages. CDK first *copies* the context into cdk.out using these patterns,
      // then Docker builds in that copy and applies .dockerignore. So .dockerignore
      // governs the image, and this governs how much gets copied to get there.
      //
      // `cdk` and `cdk.out` are the load-bearing entries: the context is the repo
      // root, so without them each synth stages the previous synth's output.
      exclude: ['cdk', 'cdk.out', '.git', '.venv', 'node_modules', 'tests', '**/__pycache__'],
    });

    this.taskRole = this.buildTaskRole(props);
    this.grantVectorDataAccess(props);

    // Built before the container environment, which includes it. The ingest trigger
    // Lambda reads the same secret, so the two sides cannot drift.
    const internalSecret = this.buildInternalSecret();
    this.containerEnvironment = {
      ...this.appEnvironment(props),
      INTERNAL_API_SECRET: internalSecret.secretValue.unsafeUnwrap(),
    };

    const cluster = new ecs.Cluster(this, 'Cluster', {
      vpc,
      containerInsightsV2: ecs.ContainerInsights.ENABLED,
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      cpu: config.appCpu,
      memoryLimitMiB: config.appMemoryMiB,
      taskRole: this.taskRole,
      runtimePlatform: {
        cpuArchitecture: ecs.CpuArchitecture.ARM64,
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
      },
    });

    taskDefinition.addContainer('api', {
      image: ecs.ContainerImage.fromDockerImageAsset(this.image),
      portMappings: [{ containerPort: APP_PORT }],
      environment: this.containerEnvironment,
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: PROJECT,
        logGroup: new logs.LogGroup(this, 'ApiLogGroup', {
          retention: logs.RetentionDays.ONE_MONTH,
        }),
      }),
      healthCheck: {
        command: ['CMD-SHELL', `python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:${APP_PORT}/health').status==200 else 1)"`],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
        startPeriod: cdk.Duration.seconds(60),
      },
    });

    // The MCP tools, as a second process on the same task.
    //
    // A separate process rather than the API's own worker, because the tool bodies are
    // `async def` with no `await` inside: their graph and Athena calls block the event loop.
    // The Retrieval agent runs in the API container and calls these over the loopback, so
    // sharing one loop would mean the agent awaiting a call that cannot be served until it
    // yields. Same image, same task role, different entrypoint.
    //
    // Not the AgentCore runtime that `McpStack` deploys: `mcp` already depends on `app` for
    // the image and the role, so pointing `app` at `mcp` would be a CloudFormation cycle.
    taskDefinition.addContainer('mcp', {
      image: ecs.ContainerImage.fromDockerImageAsset(this.image),
      portMappings: [{ containerPort: MCP_PORT }],
      environment: {
        ...this.containerEnvironment,
        APP_MODULE: 'src.mcp.server:app',
        PORT: String(MCP_PORT),
      },
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: `${PROJECT}-mcp`,
        logGroup: new logs.LogGroup(this, 'McpLogGroup', {
          retention: logs.RetentionDays.ONE_MONTH,
        }),
      }),
      // No health check: the MCP protocol has no unauthenticated GET, and a failing probe
      // would restart a task whose API half is serving fine. The agent reports its own 503.
    });

    const service = new ecs.FargateService(this, 'Service', {
      cluster,
      taskDefinition,
      desiredCount: config.appDesiredCount,
      // Private subnets with a NAT route. Not the isolated ones: Bedrock calls need
      // egress, and not public — the tasks are ALB-only.
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [appSg],
      // Roll back automatically on a failing deploy. Without this a bad image
      // sits in a crash loop until someone notices.
      circuitBreaker: { rollback: true },
      minHealthyPercent: 100,
      // The tasks are in private subnets with no inbound path except the ALB, so the
      // only way to inspect one is through SSM. Without this, diagnosing anything that
      // fails *inside* the VPC — a TLS handshake, a DNS answer, an endpoint that is
      // reachable from the account but not from the subnet — means launching a throwaway
      // task with an overridden command and reading its logs.
      enableExecuteCommand: true,
    });

    this.loadBalancer = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
      vpc,
      internetFacing: true,
      securityGroup: albSg,
    });

    const listener = this.loadBalancer.addListener('HttpListener', {
      port: 80,
      protocol: elbv2.ApplicationProtocol.HTTP,
      open: false,
    });

    listener.addTargets('ApiTarget', {
      port: APP_PORT,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [service],
      healthCheck: {
        path: '/health',
        healthyHttpCodes: '200',
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(10),
      },
      deregistrationDelay: cdk.Duration.seconds(30),
    });

    // After the ALB, because the Lambda needs its DNS name; the container environment is
    // mutated here to add the shared secret, which is why this runs before nothing else
    // reads it — `mcp` receives the same object by reference.
    this.wireIngestTrigger(props, this.loadBalancer.loadBalancerDnsName);

    new cdk.CfnOutput(this, 'AlbDnsName', { value: this.loadBalancer.loadBalancerDnsName });
    new cdk.CfnOutput(this, 'AppImageUri', { value: this.image.imageUri });
    new cdk.CfnOutput(this, 'TaskRoleArn', { value: this.taskRole.roleArn });
  }

  /**
   * OpenSearch Serverless needs BOTH an IAM grant and a data access policy naming
   * the principal. Holding only `aoss:APIAccessAll` gives a 403 with no
   * indication that the second half is missing, which is a genuinely bad hour of
   * debugging.
   *
   * It lives in `app` rather than `data` because it names the task role, and
   * declaring it next to the collection would need `data` to depend on `app`
   * while `app` already depends on `data`.
   */
  private grantVectorDataAccess(props: AppStackProps): void {
    const collectionName = props.vectorCollection.name;

    new aoss.CfnAccessPolicy(this, 'VectorDataAccessPolicy', {
      name: `${PROJECT_SLUG}-vec-data`,
      type: 'data',
      description: 'Groundwork app task role - read/write chunk embeddings',
      policy: JSON.stringify([
        {
          Rules: [
            {
              ResourceType: 'collection',
              Resource: [`collection/${collectionName}`],
              Permission: [
                'aoss:CreateCollectionItems',
                'aoss:DescribeCollectionItems',
                'aoss:UpdateCollectionItems',
              ],
            },
            {
              ResourceType: 'index',
              Resource: [`index/${collectionName}/*`],
              // No aoss:DeleteIndex. Dropping the index is a rebuild-from-S3
              // operation an operator performs deliberately, not something the
              // request-serving role should be able to do by accident.
              //
              // There is deliberately no delete permission beyond this: Serverless
              // has no `aoss:DeleteDocument`, and `aoss:WriteDocument` already
              // covers `DELETE <index>/_doc/<id>` and `_bulk`. Adding the invented
              // one failed the deploy with an unhelpful InvalidRequest naming the
              // whole policy rather than the bad value.
              Permission: [
                'aoss:CreateIndex',
                'aoss:DescribeIndex',
                'aoss:UpdateIndex',
                'aoss:ReadDocument',
                'aoss:WriteDocument',
              ],
            },
          ],
          Principal: [this.taskRole.roleArn],
        },
      ]),
    });
  }

  private appEnvironment(props: AppStackProps): Record<string, string> {
    return {
      // Load-bearing, not cosmetic. `config.validate()` treats "local" as permission to
      // relax four checks — the dev auth bypass, the model-extraction review gate, the
      // Cognito issuer requirement and the graph password requirement. A deployed task
      // defaulting to "local" is a task that would accept a config it should refuse.
      ENVIRONMENT: 'production',
      GRAPH_URI: `bolt://${props.neptuneEndpoint}:${NEPTUNE_PORT}`,
      // Neptune with IAM auth needs a signed handshake rather than a password, so
      // there is nothing here for GRAPH_PASSWORD to hold. Local dev sets it for
      // Neo4j; see .env.example.
      GRAPH_IAM_AUTH: 'true',
      GRAPH_REGION: this.region,
      VECTOR_ENDPOINT: props.vectorCollectionEndpoint,
      VECTOR_COLLECTION: props.vectorCollection.name,
      DOCUMENT_BUCKET: props.documentBucket.bucketName,
      ATHENA_RESULTS_BUCKET: props.athenaResultsBucket.bucketName,
      TENANT_TABLE: props.tenantTable.tableName,
      JOB_TABLE: props.jobTable.tableName,
      GRANT_TABLE: props.grantTable.tableName,
      COGNITO_USER_POOL_ID: props.userPoolId,
      COGNITO_CLIENT_ID: props.userPoolClientId,
      COGNITO_ISSUER_URL: props.issuerUrl,
      POLICY_STORE_ID: props.policyStoreId,
      ONTOLOGY_PACK: props.config.defaultOntology,
      AUTH_HOME_TENANT: props.config.homeTenant,
      // The loopback sidecar, not the AgentCore runtime. See the `mcp` container below.
      MCP_URL: `http://127.0.0.1:${MCP_PORT}/mcp`,
      AWS_DEFAULT_REGION: this.region,
      PAGE_BATCH_SIZE: String(PAGE_BATCH_SIZE),
      PAGE_CONCURRENCY: String(PAGE_CONCURRENCY),
      MAX_CONCURRENT_INGESTS: String(MAX_CONCURRENT_INGESTS),
    };
  }

  /**
   * S3 upload notification -> Lambda -> the API's internal ingest endpoint.
   *
   * The Lambda lives here rather than in `data` because it needs the ALB's DNS name,
   * and the notification is attached to a bucket from `data` — which works only because
   * `app` already depends on `data`, never the reverse.
   *
   * The shared secret is generated by Secrets Manager and read by both sides, so it is
   * never in the template, in git, or in a CDK context value. It is injected into the
   * task as an env var rather than fetched at runtime: the task already holds the graph
   * and the documents, so a secret it can read is not a meaningful escalation, and
   * fetching on every request would put Secrets Manager on the ingest path.
   */
  private buildInternalSecret(): secretsmanager.Secret {
    return new secretsmanager.Secret(this, 'InternalApiSecret', {
      description: 'Groundwork - shared secret for the internal ingest endpoint',
      generateSecretString: {
        passwordLength: 48,
        // The value travels in an HTTP header, so keep it to characters that need no
        // encoding and cannot be mangled by a proxy.
        excludePunctuation: true,
      },
    });
  }

  private wireIngestTrigger(props: AppStackProps, albDnsName: string): void {
    const trigger = new lambda.Function(this, 'IngestTrigger', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../lambda/ingest-trigger')),
      // It makes one HTTP call and returns; the API does the work. Anything longer
      // means the API is unreachable, and waiting will not fix that.
      timeout: cdk.Duration.seconds(15),
      memorySize: 256,
      vpc: props.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [props.appSg],
      logGroup: new logs.LogGroup(this, 'IngestTriggerLogGroup', {
        retention: logs.RetentionDays.ONE_MONTH,
      }),
      environment: {
        // HTTP, not HTTPS: the ALB listener is HTTP and this hop stays inside the VPC.
        // Put ACM on the ALB before this handles real matters.
        API_BASE_URL: `http://${albDnsName}`,
        // The same value the task holds, so the two sides cannot drift.
        INTERNAL_API_SECRET: this.containerEnvironment.INTERNAL_API_SECRET,
        RAW_PREFIX,
      },
    });

    // The bucket is referenced by *name*, not by the construct from `data`.
    //
    // This looks like indirection and is load-bearing. Attaching a notification mutates
    // the bucket's own resource, so using the construct would make `data` depend on this
    // Lambda's ARN while `app` already depends on `data` — a cycle CloudFormation
    // refuses. Importing by name puts the notification config in this stack, where it
    // belongs, and leaves the dependency one-directional.
    const bucket = s3.Bucket.fromBucketName(this, 'DocumentBucketRef', props.documentBucket.bucketName);

    // Only raw/ triggers ingestion. Without this filter the processed copy written by
    // the pipeline would re-trigger it, and every document would ingest forever.
    bucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3n.LambdaDestination(trigger),
      { prefix: RAW_PREFIX },
    );

    new cdk.CfnOutput(this, 'IngestTriggerName', { value: trigger.functionName });
  }

  /**
   * One role for the API, the workers and the MCP tools.
   *
   * Trusted by two service principals so the AgentCore runtime in `mcp` can run
   * as it directly. The alternative — a second role that assumes this one — needs
   * a trust edge from `app` back to `mcp`, and CloudFormation will not take a
   * cyclic stack dependency. This is also the honest modelling: an MCP tool must
   * not be able to do anything the REST API cannot, so it is genuinely the same
   * set of permissions rather than a copy that will drift.
   */
  private buildTaskRole(props: AppStackProps): iam.Role {
    const role = new iam.Role(this, 'TaskRole', {
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
        new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com', {
          conditions: {
            // Confused-deputy guard: without these, an AgentCore runtime in any
            // account could assume this role.
            StringEquals: { 'aws:SourceAccount': this.account },
            ArnLike: {
              'aws:SourceArn': `arn:${cdk.Aws.PARTITION}:bedrock-agentcore:${this.region}:${this.account}:*`,
            },
          },
        }),
      ),
      description: 'Groundwork API, workers, and MCP tools',
    });

    // Neptune IAM auth: scoped to this cluster's resource id, so a compromised
    // task cannot reach another cluster in the account.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['neptune-db:connect', 'neptune-db:ReadDataViaQuery', 'neptune-db:WriteDataViaQuery', 'neptune-db:DeleteDataViaQuery'],
        resources: [
          `arn:${cdk.Aws.PARTITION}:neptune-db:${this.region}:${this.account}:${props.neptuneCluster.attrClusterResourceId}/*`,
        ],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['aoss:APIAccessAll'],
        resources: [props.vectorCollection.attrArn],
      }),
    );

    props.documentBucket.grantReadWrite(role);
    props.documentKey.grantEncryptDecrypt(role);
    props.athenaResultsBucket.grantReadWrite(role);
    props.tenantTable.grantReadWriteData(role);
    props.jobTable.grantReadWriteData(role);
    props.grantTable.grantReadWriteData(role);

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['verifiedpermissions:IsAuthorized', 'verifiedpermissions:BatchIsAuthorized', 'verifiedpermissions:IsAuthorizedWithToken'],
        resources: [
          `arn:${cdk.Aws.PARTITION}:verifiedpermissions::${this.account}:policy-store/${props.policyStoreId}`,
        ],
      }),
    );

    // Bedrock model ids are not account-scoped resources, so '*' is the only
    // expressible scope here. Cost and misuse are bounded by the model allowlist
    // in src/constants.py rather than by IAM.
    //
    // This is also the only grant document reading needs: pages are transcribed by a
    // vision model rather than by an OCR service.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
        resources: ['*'],
      }),
    );

    // Read-only on the catalog. Groundwork records structured *metadata* in the
    // graph and queries rows in place, so it never needs to mutate a table.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'glue:GetDatabases',
          'glue:GetDatabase',
          'glue:GetTables',
          'glue:GetTable',
          'glue:GetPartitions',
        ],
        resources: ['*'],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'athena:StartQueryExecution',
          'athena:GetQueryExecution',
          'athena:GetQueryResults',
          'athena:StopQueryExecution',
          'athena:GetWorkGroup',
        ],
        resources: ['*'],
      }),
    );

    // Admins invite users from the app rather than the Cognito console, so the task needs
    // the admin surface — scoped to this one pool. AdminSetUserPassword stays absent:
    // Cognito mints and mails the temporary password itself, so the API never handles a
    // credential.
    //
    // Deletion is present because the product does delete users. `user_admin.delete_user`
    // removes one, and deleting a tenant removes all of them with their ownership groups.
    // An account is not what preserves the audit trail — `graph_audit` and `query_audit`
    // record what somebody did and outlive the identity that did it.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'cognito-idp:AdminGetUser',
          'cognito-idp:AdminCreateUser',
          'cognito-idp:AdminDeleteUser',
          'cognito-idp:AdminAddUserToGroup',
          'cognito-idp:AdminRemoveUserFromGroup',
          'cognito-idp:ListUsers',
          'cognito-idp:ListUsersInGroup',
          'cognito-idp:CreateGroup',
          'cognito-idp:GetGroup',
          'cognito-idp:DeleteGroup',
        ],
        resources: [
          `arn:${cdk.Aws.PARTITION}:cognito-idp:${this.region}:${this.account}:userpool/${props.userPoolId}`,
        ],
      }),
    );

    this.grantAgentCoreRuntimeAccess(role);

    return role;
  }

  /**
   * What the AgentCore runtime in `mcp` needs in order to *start*: pull the image,
   * write logs, mint workload tokens, attach ENIs. Distinct from what the tools
   * need at request time, which is everything above and shared with the API.
   *
   * Declared here rather than in `mcp` because CDK attaches a policy statement to
   * the stack owning the role. Writing it in `mcp` still emits it into the `app`
   * template, so `cdk deploy GroundworkMcp` on its own would silently not apply it.
   */
  private grantAgentCoreRuntimeAccess(role: iam.Role): void {
    this.image.repository.grantPull(role);

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          'logs:DescribeLogStreams',
        ],
        resources: [
          `arn:${cdk.Aws.PARTITION}:logs:${this.region}:${this.account}:log-group:/aws/bedrock-agentcore/*`,
        ],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'bedrock-agentcore:GetWorkloadAccessToken',
          'bedrock-agentcore:GetWorkloadAccessTokenForJWT',
        ],
        resources: ['*'],
      }),
    );

    // ENIs land in the app's own subnets and security group, so the runtime
    // reaches Neptune and the vector endpoint by exactly the rules the API has.
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'ec2:CreateNetworkInterface',
          'ec2:DescribeNetworkInterfaces',
          'ec2:DeleteNetworkInterface',
          'ec2:DescribeSubnets',
          'ec2:DescribeSecurityGroups',
          'ec2:DescribeVpcs',
        ],
        resources: ['*'],
      }),
    );
  }
}
