import { Stack, StackProps } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";
import { EnvConfig, resourceName } from "../config/constants";

export interface FoundationStackProps extends StackProps {
  envConfig: EnvConfig;
}

export class FoundationStack extends Stack {
  public readonly vpc: ec2.Vpc;
  public readonly dbSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: FoundationStackProps) {
    super(scope, id, props);

    const { envConfig } = props;

    // maxAzs triggers a context lookup (ec2:DescribeAvailabilityZones)
    // against the deploying credentials at synth/deploy time - standard CDK
    // behavior for any environment-specific stack with a VPC (the same
    // lookup `cdk init`'s generated template needs). The deploying IAM
    // principal needs that permission.
    this.vpc = new ec2.Vpc(this, "Vpc", {
      vpcName: resourceName(envConfig, "vpc"),
      maxAzs: envConfig.azCount,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: "private-isolated",
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
    });

    // S3 gateway endpoint is free and covers uploads/reports/model traffic
    // for both the backend service and the on-demand ML task.
    this.vpc.addGatewayEndpoint("S3Endpoint", {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    // Interface endpoints needed because private-isolated subnets have no
    // NAT gateway: image pulls, log shipping, and Secrets Manager reads at
    // task-start all need a path out of the VPC without public egress.
    for (const [name, service] of [
      ["EcrApiEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR],
      ["EcrDkrEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER],
      ["LogsEndpoint", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS],
      ["SecretsManagerEndpoint", ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER],
    ] as const) {
      this.vpc.addInterfaceEndpoint(name, {
        service,
        subnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      });
    }

    // dbSg lives here (not AppStack) because RDS lives in DataStack, which
    // sits between Foundation and App - but it carries no ingress rules of
    // its own. Rules that reference an App-owned security group (e.g. "allow
    // the backend service in on 5432") are added as standalone
    // CfnSecurityGroupIngress resources in AppStack instead of via
    // `.addIngressRule()` here, since the latter would make this stack
    // depend on AppStack and create a cycle (AppStack already depends on
    // Foundation for the VPC).
    this.dbSg = new ec2.SecurityGroup(this, "DbSg", {
      securityGroupName: resourceName(envConfig, "db-sg"),
      vpc: this.vpc,
      description: "RDS for PostgreSQL",
      allowAllOutbound: false,
    });
  }
}
