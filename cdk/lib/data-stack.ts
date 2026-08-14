import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as neptune from 'aws-cdk-lib/aws-neptune';
import * as aoss from 'aws-cdk-lib/aws-opensearchserverless';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

import { LexGraphConfig, NEPTUNE_PORT, PROJECT, RAW_PREFIX, tagStack } from './config';

export interface DataStackProps extends cdk.StackProps {
  readonly config: LexGraphConfig;
  readonly vpc: ec2.IVpc;
  readonly neptuneSg: ec2.ISecurityGroup;
  readonly vectorSg: ec2.ISecurityGroup;
}

/**
 * The only stack holding state that cannot be rebuilt.
 *
 * Separate from `app` on deploy cadence: Neptune takes ~15 minutes to create and
 * is touched perhaps monthly, while `app` is redeployed several times a day. It
 * is also the blast-radius boundary — a bad `app` rollback cannot reach the
 * document bucket from here.
 *
 * S3 is the source of truth. Neptune and OpenSearch are derived indexes: both can
 * be dropped and rebuilt from the bucket, which is why only the bucket is RETAINed.
 */
export class DataStack extends cdk.Stack {
  readonly documentBucket: s3.Bucket;
  readonly athenaResultsBucket: s3.Bucket;
  readonly documentKey: kms.Key;
  readonly neptuneCluster: neptune.CfnDBCluster;
  readonly neptuneEndpoint: string;
  readonly vectorCollection: aoss.CfnCollection;
  readonly vectorCollectionEndpoint: string;
  readonly tenantTable: dynamodb.Table;
  readonly jobTable: dynamodb.Table;
  readonly grantTable: dynamodb.Table;

  constructor(scope: Construct, id: string, props: DataStackProps) {
    super(scope, id, props);
    tagStack(this, 'data');

    const { config, vpc, neptuneSg, vectorSg } = props;

    this.documentKey = this.buildDocumentKey();
    this.documentBucket = this.buildDocumentBucket();
    this.athenaResultsBucket = this.buildAthenaResultsBucket();

    const { cluster, endpoint } = this.buildNeptune(config, vpc, neptuneSg);
    this.neptuneCluster = cluster;
    this.neptuneEndpoint = endpoint;

    const { collection, endpoint: vectorEndpoint } = this.buildVectorStore(config, vpc, vectorSg);
    this.vectorCollection = collection;
    this.vectorCollectionEndpoint = vectorEndpoint;

    const tables = this.buildTables();
    this.tenantTable = tables.tenants;
    this.jobTable = tables.jobs;
    this.grantTable = tables.grants;

    this.emitOutputs();
  }

  private buildDocumentKey(): kms.Key {
    return new kms.Key(this, 'DocumentKey', {
      description: 'LexGraph source documents - customer-managed so key access is auditable',
      enableKeyRotation: true,
      // RETAIN, not DESTROY: destroying the key makes every object in a RETAINed
      // bucket permanently unreadable, which is a data-loss event dressed up as
      // a clean teardown.
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });
  }

  /**
   * Object Lock is deliberately OFF. It can only be set at bucket creation, so the
   * decision is permanent for a given bucket — see cdk/README.md for turning it on
   * in production.
   *
   * Two reasons it cannot be the default here. Uploads arrive by presigned POST
   * under `raw/`, so a 365-day GOVERNANCE retention would apply to every abandoned
   * or malicious upload and need s3:BypassGovernanceRetention to clear a typo. And
   * the processed key is content-addressed, so the final key is unknowable until
   * the bytes have been hashed — the object that lands first is not the object that
   * is the record.
   */
  private buildDocumentBucket(): s3.Bucket {
    const webOrigin = this.node.tryGetContext('webOrigin') as string | undefined;

    return new s3.Bucket(this, 'DocumentBucket', {
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: this.documentKey,
      bucketKeyEnabled: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      // The browser POSTs the file straight to S3, so the upload never crosses the
      // API and is not bounded by the 60s CloudFront origin timeout. Until
      // `webOrigin` is set on the second pass there is no origin to allow, and an
      // empty list is correct rather than a wildcard.
      cors: webOrigin
        ? [
            {
              allowedOrigins: [webOrigin],
              allowedMethods: [s3.HttpMethods.POST],
              allowedHeaders: ['*'],
              exposedHeaders: ['ETag'],
              maxAge: 3000,
            },
          ]
        : [],
      lifecycleRules: [
        {
          // Assertions cite page and char offsets into a specific version, so old
          // versions stay readable — just cheaply.
          noncurrentVersionTransitions: [
            { storageClass: s3.StorageClass.INFREQUENT_ACCESS, transitionAfter: cdk.Duration.days(90) },
          ],
          abortIncompleteMultipartUploadAfter: cdk.Duration.days(7),
        },
        {
          // A raw/ object is consumed within seconds of landing and the processed
          // copy is the record. Anything still here days later was abandoned or
          // failed every retry, and keeping it serves nobody.
          id: 'expire-unprocessed-uploads',
          prefix: RAW_PREFIX,
          expiration: cdk.Duration.days(7),
        },
      ],
    });
  }

  private buildAthenaResultsBucket(): s3.Bucket {
    return new s3.Bucket(this, 'AthenaResultsBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      // Query results are a cache of rows that live in someone else's warehouse.
      // Nothing here is a source of truth, so it is disposable by design.
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      lifecycleRules: [{ expiration: cdk.Duration.days(14) }],
    });
  }

  /**
   * Neptune Database (provisioned), not Neptune Analytics.
   *
   * Analytics has a 32 m-NCU floor per graph and no autoscaling, so a
   * multi-tenant SaaS with dozens of small tenants would pay for 32 m-NCU per
   * tenant. Neptune Database gives us openCypher over Bolt on one cluster with
   * one property graph, which is what `src/graph/scope.py` assumes.
   *
   * ONE property graph per cluster is a hard Neptune constraint, not a choice:
   * named graphs are RDF/SPARQL only and openCypher cannot address them. So
   * tenancy is a property filter, and `scope.py` is the only thing enforcing it.
   */
  private buildNeptune(
    config: LexGraphConfig,
    vpc: ec2.IVpc,
    neptuneSg: ec2.ISecurityGroup,
  ): { cluster: neptune.CfnDBCluster; endpoint: string } {
    const subnetGroup = new neptune.CfnDBSubnetGroup(this, 'NeptuneSubnets', {
      dbSubnetGroupDescription: 'LexGraph Neptune - isolated subnets',
      subnetIds: vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_ISOLATED }).subnetIds,
    });

    // The parameter group family must match the engine's major line, and the
    // two are set in one place because a mismatch is a create-time failure:
    // a neptune1.3 group on a 1.4.x cluster is rejected outright. We do not pin
    // engineVersion — Neptune's default is already on the 1.4 line, and pinning a
    // specific minor means this template breaks the day that minor is retired.
    const clusterParams = new neptune.CfnDBClusterParameterGroup(this, 'NeptuneClusterParams', {
      family: config.neptuneParameterGroupFamily,
      description: 'LexGraph - audit logging on',
      parameters: {
        // Every write to the graph is an assertion about a customer's legal
        // matter. Audit logs are how we answer "who wrote this edge" when the
        // provenance chain itself is what is under question.
        neptune_enable_audit_log: '1',
      },
    });

    const cluster = new neptune.CfnDBCluster(this, 'Neptune', {
      dbSubnetGroupName: subnetGroup.ref,
      dbClusterParameterGroupName: clusterParams.ref,
      vpcSecurityGroupIds: [neptuneSg.securityGroupId],
      dbPort: NEPTUNE_PORT,
      storageEncrypted: true,
      // IAM auth rather than a password: the app assumes a task role already, so
      // there is no Neptune credential to leak or rotate.
      iamAuthEnabled: true,
      backupRetentionPeriod: 7,
      enableCloudwatchLogsExports: ['audit'],
      copyTagsToSnapshot: true,
      // Off for now so `cdk destroy` works during development. Turn this on
      // before the first real tenant — see cdk/README.md.
      deletionProtection: false,
    });
    cluster.addResourceDependency(subnetGroup);
    cluster.addResourceDependency(clusterParams);
    cluster.applyRemovalPolicy(cdk.RemovalPolicy.SNAPSHOT);

    // A cluster with no instances accepts no queries. Writer is instance 0;
    // raising neptuneInstanceCount adds read replicas.
    for (let i = 0; i < config.neptuneInstanceCount; i++) {
      const instance = new neptune.CfnDBInstance(this, `NeptuneInstance${i}`, {
        dbClusterIdentifier: cluster.ref,
        // db.t4g.medium is free-tier eligible for 750 hours and Graviton, so
        // cheaper than t3. It is explicitly NOT for production: 2:1 RAM-to-vCPU
        // disables the DFE statistics that make openCypher fast, and large
        // traversals will OOM. Deliberate for now — raise via context, no code
        // edit, before any load test.
        dbInstanceClass: config.neptuneInstanceClass,
        dbSubnetGroupName: subnetGroup.ref,
        autoMinorVersionUpgrade: true,
      });
      instance.addResourceDependency(cluster);
      instance.addResourceDependency(subnetGroup);
    }

    return { cluster, endpoint: cluster.attrEndpoint };
  }

  /**
   * OpenSearch Serverless VECTORSEARCH collection inside a NextGen collection
   * group whose minimum OCU is 0.
   *
   * The collection group is the only place scale-to-zero can be configured, and
   * it is worth the extra resource: OCUs are the largest recurring line in this
   * whole stack, and an idle dev collection at the CLASSIC 2-OCU floor costs more
   * per month than the Neptune instance. Max OCU is capped to bound the downside.
   */
  private buildVectorStore(
    config: LexGraphConfig,
    vpc: ec2.IVpc,
    vectorSg: ec2.ISecurityGroup,
  ): { collection: aoss.CfnCollection; endpoint: string } {
    const collectionName = `${PROJECT}-vectors`;

    const group = new aoss.CfnCollectionGroup(this, 'VectorCollectionGroup', {
      name: `${PROJECT}-vector-group`,
      generation: 'NEXTGEN',
      // ENABLED even though this is a dev-sized deployment. Normally standby
      // replicas double the OCU floor, but with a minimum of 0 there is no floor
      // to double — an idle group bills nothing either way, so the redundancy is
      // free. Every AWS NextGen scale-to-zero example also pairs the two.
      standbyReplicas: 'ENABLED',
      capacityLimits: {
        minIndexingCapacityInOcu: config.vectorMinOcu,
        minSearchCapacityInOcu: config.vectorMinOcu,
        maxIndexingCapacityInOcu: config.vectorMaxOcu,
        maxSearchCapacityInOcu: config.vectorMaxOcu,
      },
    });

    const encryptionPolicy = new aoss.CfnSecurityPolicy(this, 'VectorEncryptionPolicy', {
      name: `${PROJECT}-vec-enc`,
      type: 'encryption',
      policy: JSON.stringify({
        Rules: [{ ResourceType: 'collection', Resource: [`collection/${collectionName}`] }],
        AWSOwnedKey: true,
      }),
    });

    // The chunk text in this index is verbatim from privileged documents, so the
    // collection is never reachable from the public internet — only through this
    // endpoint, referenced by ID in the network policy below.
    const vpcEndpoint = new aoss.CfnVpcEndpoint(this, 'VectorVpcEndpoint', {
      name: `${PROJECT}-vec-vpce`,
      vpcId: vpc.vpcId,
      subnetIds: vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnetIds,
      securityGroupIds: [vectorSg.securityGroupId],
    });

    const networkPolicy = new aoss.CfnSecurityPolicy(this, 'VectorNetworkPolicy', {
      name: `${PROJECT}-vec-net`,
      type: 'network',
      policy: JSON.stringify([
        {
          Rules: [
            { ResourceType: 'collection', Resource: [`collection/${collectionName}`] },
            // Dashboards deliberately absent: there is no read path to
            // privileged text that bypasses tenant scoping.
          ],
          AllowFromPublic: false,
          SourceVPCEs: [vpcEndpoint.ref],
        },
      ]),
    });

    const collection = new aoss.CfnCollection(this, 'VectorCollection', {
      name: collectionName,
      type: 'VECTORSEARCH',
      // Membership of the group is what makes scale-to-zero apply; a collection
      // outside a group falls back to the account-level minimum, which is not 0.
      collectionGroupName: group.name,
      standbyReplicas: 'ENABLED',
      description: 'LexGraph document chunk embeddings',
    });
    // Both policies must exist before the collection or creation fails with an
    // unhelpful "no matching security policy".
    collection.addResourceDependency(encryptionPolicy);
    collection.addResourceDependency(networkPolicy);
    collection.addResourceDependency(group);
    collection.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    return { collection, endpoint: collection.attrCollectionEndpoint };
  }

  /**
   * Control-plane tables. All three are small, spiky, and read by key — which is
   * exactly what on-demand billing is for. None of them holds graph data.
   */
  private buildTables(): {
    tenants: dynamodb.Table;
    jobs: dynamodb.Table;
    grants: dynamodb.Table;
  } {
    const common = {
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
    };

    // Who belongs to which tenant. Keys are owned by `src/tenant_directory.py`:
    //   PK = USER#{cognito sub}    SK = PROFILE
    //   GSI1PK = EMAIL#{lowercased email}
    //
    // Single-table rather than one row per tenant, because the question actually asked on
    // every request is "which tenant is this subject in" — and the answer must be a
    // get_item, not a scan, since it sits on the authentication path.
    //
    // Keyed on the Cognito `sub`, not email: an email can be reassigned to a different
    // person, a sub cannot. The email is indexed separately so an operator can still find
    // a record by the identifier they know.
    const tenants = new dynamodb.Table(this, 'TenantTable', {
      ...common,
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
      // RETAIN: this table is the only mapping from a verified identity to a tenant.
      // Losing it orphans every S3 prefix and every graph node.
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    tenants.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKey: { name: 'GSI1PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'GSI1SK', type: dynamodb.AttributeType.STRING },
    });

    // Ingest job state. Keys are owned by `src/documents/job_store.py`:
    //   PK = TENANT#{t}    SK = JOB#{job_id}
    //   GSI1PK = TENANT#{t}#DOC#{document_id}
    const jobs = new dynamodb.Table(this, 'JobTable', {
      ...common,
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
      // Ingestion runs are replayable from S3, so expiring their bookkeeping
      // costs nothing but a re-run.
      timeToLiveAttribute: 'expires_at',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // The UI polls by document id, because that is what the browser holds after an
    // upload — it has no job id until the notification has been processed. Without
    // this index that poll would be a scan.
    jobs.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKey: { name: 'GSI1PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'GSI1SK', type: dynamodb.AttributeType.STRING },
    });

    // Matter assignments, ethical screens, and the append-only audit trail, in one
    // table. Keys are owned by `src/access_dynamo.py`:
    //   PK = TENANT#{t}#USER#{u}    SK = ASSIGN#{m} | SCREEN#{m} | EVENT#{at}#{uuid}
    // RETAIN because this is the only thing here that is not rebuildable. Neptune and
    // the vector index are derived from S3; the record of who screened whom, when and
    // why exists nowhere else, and it is the compliance artifact.
    const grants = new dynamodb.Table(this, 'GrantTable', {
      ...common,
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // "Who is on this matter" is asked on an authorization path, so it must be a
    // query. A scan would slow with tenant size and eventually time out — a store that
    // fails under load is not one to put an ethical wall behind.
    grants.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKey: { name: 'GSI1PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'GSI1SK', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    return { tenants, jobs, grants };
  }

  private emitOutputs(): void {
    new cdk.CfnOutput(this, 'NeptuneEndpoint', {
      value: this.neptuneEndpoint,
      description: `Writer endpoint - bolt://<endpoint>:${NEPTUNE_PORT}`,
    });
    new cdk.CfnOutput(this, 'VectorCollectionEndpoint', {
      value: this.vectorCollectionEndpoint,
    });
    new cdk.CfnOutput(this, 'DocumentBucketName', { value: this.documentBucket.bucketName });
    new cdk.CfnOutput(this, 'AthenaResultsBucketName', {
      value: this.athenaResultsBucket.bucketName,
    });
  }
}
