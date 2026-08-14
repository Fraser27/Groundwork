import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

import { APP_PORT, LexGraphConfig, NEPTUNE_PORT, tagStack } from './config';

export interface NetworkStackProps extends cdk.StackProps {
  readonly config: LexGraphConfig;
}

/**
 * AZ IDs that BOTH AgentCore Runtime VPC connectivity and the OpenSearch
 * Serverless data-plane endpoint are offered in, per region.
 *
 * This exists because the two services publish *different* subsets, and the
 * intersection is smaller than either. AgentCore in us-east-1 supports
 * use1-az1, use1-az2 and use1-az4 only; put a subnet in use1-az3 or use1-az5 and
 * the runtime fails to create, with an error that names the subnet rather than
 * the AZ. Nothing in a synthesised template records why the AZ list is short, so
 * the next person to "simplify" the network stack by dropping `availabilityZones`
 * and letting CDK pick will break `mcp` and `data` and have no idea why.
 *
 * Note these are AZ *IDs*, not names. `us-east-1a` maps to a different physical
 * zone in every account, so the mapping has to be resolved per account:
 *
 *   aws ec2 describe-availability-zones \
 *     --query 'AvailabilityZones[].[ZoneName,ZoneId]' --output text
 *
 * then set `availabilityZones` in cdk.json to the matching NAMES. Until that is
 * done we fall back to CDK's default two AZs, which is fine for `network` and
 * `data` but is the thing to check first if `mcp` will not deploy.
 */
export const SUPPORTED_AZ_IDS: Record<string, string[]> = {
  'us-east-1': ['use1-az1', 'use1-az2', 'use1-az4'],
  'us-east-2': ['use2-az1', 'use2-az2', 'use2-az3'],
  'us-west-2': ['usw2-az1', 'usw2-az2', 'usw2-az3'],
  'eu-west-1': ['euw1-az1', 'euw1-az2', 'euw1-az3'],
  'eu-central-1': ['euc1-az1', 'euc1-az2', 'euc1-az3'],
};

export class NetworkStack extends cdk.Stack {
  readonly vpc: ec2.Vpc;

  /** Anything that needs to reach Neptune or OpenSearch. */
  readonly appSg: ec2.SecurityGroup;
  readonly neptuneSg: ec2.SecurityGroup;
  readonly vectorSg: ec2.SecurityGroup;
  readonly albSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: NetworkStackProps) {
    super(scope, id, props);
    tagStack(this, 'network');

    const { config } = props;

    this.vpc = new ec2.Vpc(this, 'Vpc', {
      ipAddresses: ec2.IpAddresses.cidr('10.20.0.0/16'),
      ...(config.availabilityZones
        ? { availabilityZones: config.availabilityZones }
        : { maxAzs: 2 }),
      // One NAT gateway, not one per AZ. It is a single point of failure for
      // egress and ~$33/mo; both are acceptable pre-production, and the fix is
      // this one number.
      natGateways: 1,
      subnetConfiguration: [
        { name: 'public', subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: 'private', subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 24 },
        // Neptune gets isolated subnets: it has no reason to reach the internet,
        // and it holds the only state in the system that cannot be rebuilt.
        { name: 'data', subnetType: ec2.SubnetType.PRIVATE_ISOLATED, cidrMask: 24 },
      ],
    });

    this.albSg = new ec2.SecurityGroup(this, 'AlbSg', {
      vpc: this.vpc,
      // CloudFormation restricts security group descriptions to a limited ASCII
      // set, so no em dashes here even though the rest of the file uses them.
      description: 'ALB - HTTP from CloudFront/internet',
      allowAllOutbound: true,
    });
    this.albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), 'HTTP');

    this.appSg = new ec2.SecurityGroup(this, 'AppSg', {
      vpc: this.vpc,
      description: 'Fargate tasks, workers, and the MCP runtime',
      allowAllOutbound: true,
    });
    this.appSg.addIngressRule(this.albSg, ec2.Port.tcp(APP_PORT), 'API from ALB');

    this.neptuneSg = new ec2.SecurityGroup(this, 'NeptuneSg', {
      vpc: this.vpc,
      description: 'Neptune cluster - Bolt/openCypher from the app only',
      // Nothing in the data tier initiates outbound connections, so denying it
      // makes an exfiltration path one rule further away.
      allowAllOutbound: false,
    });
    this.neptuneSg.addIngressRule(this.appSg, ec2.Port.tcp(NEPTUNE_PORT), 'openCypher from app');

    this.vectorSg = new ec2.SecurityGroup(this, 'VectorSg', {
      vpc: this.vpc,
      description: 'OpenSearch Serverless VPC endpoint ENIs',
      allowAllOutbound: false,
    });
    this.vectorSg.addIngressRule(this.appSg, ec2.Port.tcp(443), 'HTTPS from app');

    this.addEndpoints();

    new cdk.CfnOutput(this, 'VpcId', { value: this.vpc.vpcId });
    new cdk.CfnOutput(this, 'AzNames', {
      value: this.vpc.availabilityZones.join(','),
      description: 'Cross-check these against SUPPORTED_AZ_IDS in network-stack.ts',
    });
  }

  /**
   * Interface endpoints for everything the app calls, so the private subnets can
   * work without routing AWS API traffic through NAT.
   *
   * S3 is a gateway endpoint — free, and documents are the largest byte volume in
   * the system, so this is the one endpoint that pays for itself immediately.
   * The interface endpoints are ~$7/mo each per AZ, which is the trade for not
   * paying NAT data-processing on every Bedrock call. Page transcription is a
   * Bedrock vision call, so Bedrock now carries the whole document-reading path.
   */
  private addEndpoints(): void {
    this.vpc.addGatewayEndpoint('S3Gateway', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
      subnets: [
        { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      ],
    });

    this.vpc.addGatewayEndpoint('DynamoDbGateway', {
      service: ec2.GatewayVpcEndpointAwsService.DYNAMODB,
      subnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
    });

    const interfaceEndpoints: Record<string, ec2.InterfaceVpcEndpointAwsService> = {
      Bedrock: ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
      Athena: ec2.InterfaceVpcEndpointAwsService.ATHENA,
      Glue: ec2.InterfaceVpcEndpointAwsService.GLUE,
      Sts: ec2.InterfaceVpcEndpointAwsService.STS,
      Secrets: ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
      // ECR pair plus CloudWatch Logs: without all three, Fargate tasks in a
      // private subnet cannot pull an image or report why they died.
      EcrApi: ec2.InterfaceVpcEndpointAwsService.ECR,
      EcrDocker: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
      Logs: ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
    };

    for (const [name, service] of Object.entries(interfaceEndpoints)) {
      this.vpc.addInterfaceEndpoint(`${name}Endpoint`, {
        service,
        subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        securityGroups: [this.appSg],
        privateDnsEnabled: true,
      });
    }
  }
}
