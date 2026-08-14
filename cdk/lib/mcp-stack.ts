import * as cdk from 'aws-cdk-lib';
import * as agentcore from 'aws-cdk-lib/aws-bedrockagentcore';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

import { APP_PORT, PROJECT, tagStack } from './config';

export interface McpStackProps extends cdk.StackProps {
  readonly vpc: ec2.IVpc;
  readonly appSg: ec2.ISecurityGroup;
  /** Built by `app` and reused verbatim — see the class docstring. */
  readonly image: ecrAssets.DockerImageAsset;
  /**
   * The same role the API runs as, so a tool cannot exceed what the API can do.
   * `app` already declares `bedrock-agentcore.amazonaws.com` in its trust policy;
   * the runtime-specific grants are added to it here, where they are used.
   */
  readonly taskRole: iam.IRole;
  /** Built by `app`; this stack overrides only APP_MODULE. */
  readonly containerEnvironment: Record<string, string>;
  readonly userPoolClientId: string;
  readonly issuerUrl: string;
}

/**
 * The MCP server on Bedrock AgentCore Runtime.
 *
 * Runs the *same image* as `app`, entered at the MCP module instead of the API
 * module. That is the important decision: an MCP tool that answered a question
 * differently from the REST endpoint would mean two governance implementations,
 * and the tools would be the one nobody audits. Same image, same `scope.py`, same
 * ontology gate.
 *
 * AgentCore's JWT authorizer validates the Cognito token before our code runs, so
 * a tool call arrives with the same verified `tenant_id` claim an API request
 * does. There is no service-account path into the graph.
 *
 * Deployment note: this stack must land in AZs AgentCore supports. See
 * SUPPORTED_AZ_IDS in network-stack.ts — that list is the reason this deploys.
 */
export class McpStack extends cdk.Stack {
  readonly runtime: agentcore.CfnRuntime;

  constructor(scope: Construct, id: string, props: McpStackProps) {
    super(scope, id, props);
    tagStack(this, 'mcp');

    const { vpc, appSg, image, taskRole } = props;

    // Note: the IAM grants this runtime needs are declared in `app`, next to the
    // role, not here. CDK attaches a policy statement to the stack that owns the
    // role regardless of where it is written, so declaring them here would put
    // them in the `app` template anyway — and `cdk deploy LexGraphMcp` alone
    // would then not apply them. See buildTaskRole in app-stack.ts.

    this.runtime = new agentcore.CfnRuntime(this, 'McpRuntime', {
      agentRuntimeName: `${PROJECT}_mcp`,
      description: 'LexGraph governed semantic layer - MCP tools',
      roleArn: taskRole.roleArn,
      agentRuntimeArtifact: {
        containerConfiguration: { containerUri: image.imageUri },
      },
      // MCP, not HTTP: the runtime then speaks the streamable-HTTP MCP transport
      // and any MCP client can connect without a shim.
      protocolConfiguration: 'MCP',
      networkConfiguration: {
        networkMode: 'VPC',
        // No vpcId field: AgentCore derives the VPC from the subnets, which is
        // also why the subnets must be in a supported AZ or creation fails with
        // an error that names the subnet and not the zone.
        networkModeConfig: {
          subnets: vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnetIds,
          securityGroups: [appSg.securityGroupId],
        },
      },
      authorizerConfiguration: {
        customJwtAuthorizer: {
          discoveryUrl: `${props.issuerUrl}/.well-known/openid-configuration`,
          allowedClients: [props.userPoolClientId],
        },
      },
      environmentVariables: {
        ...props.containerEnvironment,
        // The one thing that differs from the API container: which module runs.
        APP_MODULE: 'src.mcp.server:app',
        PORT: String(APP_PORT),
      },
    });

    const endpoint = new agentcore.CfnRuntimeEndpoint(this, 'McpEndpoint', {
      agentRuntimeId: this.runtime.attrAgentRuntimeId,
      name: 'live',
      description: 'Stable endpoint - clients point here, versions move beneath it',
    });
    endpoint.addResourceDependency(this.runtime);

    new cdk.CfnOutput(this, 'McpRuntimeArn', { value: this.runtime.attrAgentRuntimeArn });
    new cdk.CfnOutput(this, 'McpEndpointArn', { value: endpoint.attrAgentRuntimeEndpointArn });
  }
}
