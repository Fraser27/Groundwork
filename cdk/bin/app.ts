#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';

import { PROJECT, PROJECT_SLUG, readConfig } from '../lib/config';
import { NetworkStack } from '../lib/network-stack';
import { DataStack } from '../lib/data-stack';
import { AuthStack } from '../lib/auth-stack';
import { AppStack } from '../lib/app-stack';
import { McpStack } from '../lib/mcp-stack';
import { WebStack } from '../lib/web-stack';

const app = new cdk.App();
const config = readConfig(app);

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
};

const prefix = 'Groundwork';

/**
 * Six stacks, split by deploy cadence and blast radius rather than by feature.
 *
 * `data` is separate because Neptune takes ~15 minutes to create and holds the
 * only state that cannot be rebuilt from S3. `app` is separate because it is
 * redeployed several times a day, and a rollback there should not be able to take
 * a CloudFormation lock on the graph.
 *
 * Everything is wired by construct reference, not by CfnOutput/Fn::ImportValue, so
 * a bad `app` deploy cannot be blocked by a stale export. The cost is that CDK
 * infers the dependency order and stacks must be deployed together the first time.
 */
const network = new NetworkStack(app, `${prefix}Network`, {
  env,
  config,
  description: 'Groundwork — VPC, subnets, security groups, VPC endpoints',
});

const data = new DataStack(app, `${prefix}Data`, {
  env,
  config,
  vpc: network.vpc,
  neptuneSg: network.neptuneSg,
  description: 'Groundwork — Neptune, OpenSearch Serverless, DynamoDB, S3',
});

/**
 * The one genuinely circular requirement: the Cognito hosted UI needs the
 * CloudFront domain as a callback URL, and CloudFront needs the ALB, which needs
 * Cognito's issuer. Broken with context rather than a custom resource — deploy
 * once, read the WebUrl output, then set `webOrigin` in cdk.json and redeploy
 * `auth`. Two passes, but no Lambda whose failure mode is a half-configured
 * login page.
 */
const webOrigin = app.node.tryGetContext('webOrigin') as string | undefined;

const auth = new AuthStack(app, `${prefix}Auth`, {
  env,
  extraCallbackUrls: webOrigin ? [`${webOrigin}/`] : [],
  description: 'Groundwork — Cognito user pool, hosted UI, Cedar policy store',
});

const appStack = new AppStack(app, `${prefix}App`, {
  env,
  config,
  vpc: network.vpc,
  appSg: network.appSg,
  albSg: network.albSg,
  neptuneCluster: data.neptuneCluster,
  neptuneEndpoint: data.neptuneEndpoint,
  vectorCollection: data.vectorCollection,
  vectorCollectionEndpoint: data.vectorCollectionEndpoint,
  documentBucket: data.documentBucket,
  athenaResultsBucket: data.athenaResultsBucket,
  documentKey: data.documentKey,
  tenantTable: data.tenantTable,
  jobTable: data.jobTable,
  grantTable: data.grantTable,
  userPoolId: auth.userPool.userPoolId,
  userPoolClientId: auth.userPoolClient.userPoolClientId,
  issuerUrl: auth.issuerUrl,
  policyStoreId: auth.policyStore.attrPolicyStoreId,
  description: 'Groundwork — FastAPI on Fargate behind an ALB',
});

// Opt-in, because it is the only reason the image has to be ARM64 (see
// GroundworkConfig.agentCoreMcp). Not deploying it leaves the MCP tools running as a
// sidecar for the Retrieval agent; what is lost is the authenticated endpoint an
// outside MCP client would connect to.
if (config.agentCoreMcp) {
  new McpStack(app, `${prefix}Mcp`, {
    env,
    vpc: network.vpc,
    appSg: network.appSg,
    image: appStack.image,
    taskRole: appStack.taskRole,
    containerEnvironment: appStack.containerEnvironment,
    userPoolClientId: auth.userPoolClient.userPoolClientId,
    issuerUrl: auth.issuerUrl,
    description: 'Groundwork — MCP server on Bedrock AgentCore Runtime',
  });
}

new WebStack(app, `${prefix}Web`, {
  env,
  loadBalancer: appStack.loadBalancer,
  userPoolId: auth.userPool.userPoolId,
  userPoolClientId: auth.userPoolClient.userPoolClientId,
  hostedUiDomain: `${PROJECT_SLUG}-${cdk.Aws.ACCOUNT_ID}.auth.${env.region}.amazoncognito.com`,
  description: 'Groundwork — CloudFront + S3 for the React UI',
});

// Applied at the app level so it reaches every stack, including any added later
// without remembering to tag it.
cdk.Tags.of(app).add('Project', PROJECT);
