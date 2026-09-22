import { CfnOutput, Duration, RemovalPolicy, SecretValue, Stack, StackProps } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecsPatterns from "aws-cdk-lib/aws-ecs-patterns";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as rds from "aws-cdk-lib/aws-rds";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as amplify from "@aws-cdk/aws-amplify-alpha";
import { Construct } from "constructs";
import {
  BACKEND_CONTAINER_PORT,
  BACKEND_CPU_TARGET_UTILIZATION_PERCENT,
  BACKEND_MAX_TASK_COUNT,
  BACKEND_MIN_TASK_COUNT,
  BACKEND_TASK_CPU,
  BACKEND_TASK_MEMORY_MIB,
  ECR_MAX_IMAGE_COUNT,
  EnvConfig,
  ML_TASK_CPU,
  ML_TASK_MEMORY_MIB,
  resourceName,
} from "../config/constants";

export interface AppStackProps extends StackProps {
  envConfig: EnvConfig;
  vpc: ec2.IVpc;
  dbSg: ec2.ISecurityGroup;
  dbInstance: rds.DatabaseInstance;
  bucket: s3.Bucket;
  userPool: cognito.UserPool;
  userPoolClient: cognito.UserPoolClient;
}

export class AppStack extends Stack {
  constructor(scope: Construct, id: string, props: AppStackProps) {
    super(scope, id, props);

    const { envConfig, vpc, dbSg, dbInstance, bucket, userPool, userPoolClient } = props;

    // backendServiceSg/mlTaskSg are created here (not FoundationStack)
    // because ecs_patterns.ApplicationLoadBalancedFargateService wires an
    // ALB -> service ingress rule onto the service's security group
    // automatically - that rule has to live in the same stack as the ALB,
    // or CDK can't express the dependency without a cycle.
    const backendServiceSg = new ec2.SecurityGroup(this, "BackendServiceSg", {
      securityGroupName: resourceName(envConfig, "backend-service-sg"),
      vpc,
      description: "Backend FastAPI Fargate service",
      allowAllOutbound: true,
    });

    const mlTaskSg = new ec2.SecurityGroup(this, "MlTaskSg", {
      securityGroupName: resourceName(envConfig, "ml-task-sg"),
      vpc,
      description: "On-demand ML inference Fargate task (RunTask, no Service)",
      allowAllOutbound: true,
    });

    // Standalone ingress resource (not dbSg.addIngressRule(...)) so the rule
    // lives in this stack rather than FoundationStack, which owns dbSg -
    // AppStack already depends on Foundation for the VPC, so this direction
    // is safe; the reverse (Foundation referencing an App-owned SG) is not.
    // Intentionally no equivalent rule for mlTaskSg: batch_predict.py is
    // S3-only and never needs to reach RDS.
    new ec2.CfnSecurityGroupIngress(this, "DbIngressFromBackend", {
      groupId: dbSg.securityGroupId,
      sourceSecurityGroupId: backendServiceSg.securityGroupId,
      ipProtocol: "tcp",
      fromPort: 5432,
      toPort: 5432,
      description: "Backend service reads/writes Postgres directly",
    });

    const backendRepo = new ecr.Repository(this, "BackendRepo", {
      repositoryName: resourceName(envConfig, "backend"),
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    backendRepo.addLifecycleRule({ maxImageCount: ECR_MAX_IMAGE_COUNT });

    const mlWorkerRepo = new ecr.Repository(this, "MlWorkerRepo", {
      repositoryName: resourceName(envConfig, "ml-worker"),
      imageScanOnPush: true,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    mlWorkerRepo.addLifecycleRule({ maxImageCount: ECR_MAX_IMAGE_COUNT });

    const cluster = new ecs.Cluster(this, "Cluster", {
      clusterName: resourceName(envConfig, "cluster"),
      vpc,
      containerInsightsV2: ecs.ContainerInsights.ENABLED,
    });

    // --- ML task definition: registered for on-demand ecs:RunTask only ---
    // No FargateService/ALB attachment - this task is started directly by
    // the backend's background asyncio task when a reviewer approves an
    // ingestion job, and exits when the run completes.
    const mlTaskDefinition = new ecs.FargateTaskDefinition(this, "MlTaskDefinition", {
      family: resourceName(envConfig, "ml-task"),
      cpu: ML_TASK_CPU,
      memoryLimitMiB: ML_TASK_MEMORY_MIB,
    });
    mlTaskDefinition.addContainer("MlWorker", {
      containerName: "ml-worker",
      image: ecs.ContainerImage.fromEcrRepository(mlWorkerRepo, "latest"),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: resourceName(envConfig, "ml-task") }),
      environment: {
        S3_BUCKET: bucket.bucketName,
        AWS_REGION: this.region,
      },
    });
    bucket.grantReadWrite(mlTaskDefinition.taskRole);
    // ML task has no DB access: batch_predict.py is S3-only, the backend
    // does all DB writes after RunTask completes and results are read back
    // from S3 - so no DB secret/security-group rule is granted here.

    // --- Backend service: always-on ApplicationLoadBalancedFargateService ---
    const backendService = new ecsPatterns.ApplicationLoadBalancedFargateService(
      this,
      "BackendService",
      {
        serviceName: resourceName(envConfig, "backend"),
        cluster,
        cpu: BACKEND_TASK_CPU,
        memoryLimitMiB: BACKEND_TASK_MEMORY_MIB,
        desiredCount: BACKEND_MIN_TASK_COUNT,
        publicLoadBalancer: true,
        taskSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
        securityGroups: [backendServiceSg],
        // Plain HTTP for now: an HTTPS listener needs an ACM certificate,
        // which needs a domain name that hasn't been chosen yet. Once one
        // is registered, add `certificate`/`redirectHTTP: true` and switch
        // this to listenerPort: 443.
        listenerPort: 80,
        taskImageOptions: {
          image: ecs.ContainerImage.fromEcrRepository(backendRepo, "latest"),
          containerPort: BACKEND_CONTAINER_PORT,
          logDriver: ecs.LogDrivers.awsLogs({ streamPrefix: resourceName(envConfig, "backend") }),
          secrets: {
            DATABASE_URL: ecs.Secret.fromSecretsManager(dbInstance.secret!),
          },
          environment: {
            AWS_REGION: this.region,
            S3_BUCKET: bucket.bucketName,
            COGNITO_USER_POOL_ID: userPool.userPoolId,
            COGNITO_CLIENT_ID: userPoolClient.userPoolClientId,
            COGNITO_ISSUER_URL: `https://cognito-idp.${this.region}.amazonaws.com/${userPool.userPoolId}`,
            ECS_CLUSTER_ARN: cluster.clusterArn,
            ML_TASK_DEFINITION_ARN: mlTaskDefinition.taskDefinitionArn,
            ML_TASK_SUBNET_IDS: vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_ISOLATED })
              .subnetIds.join(","),
            ML_TASK_SECURITY_GROUP_ID: mlTaskSg.securityGroupId,
          },
        },
      },
    );
    backendService.targetGroup.configureHealthCheck({ path: "/health" });

    backendService.service.autoScaleTaskCount({
      minCapacity: BACKEND_MIN_TASK_COUNT,
      maxCapacity: BACKEND_MAX_TASK_COUNT,
    }).scaleOnCpuUtilization("CpuScaling", {
      targetUtilizationPercent: BACKEND_CPU_TARGET_UTILIZATION_PERCENT,
      scaleInCooldown: Duration.seconds(60),
      scaleOutCooldown: Duration.seconds(60),
    });

    // Backend task role permissions to drive the on-demand ML task: RunTask
    // and PassRole are scoped to exactly the ML task definition/roles (not
    // "*") to avoid a broad PassRole privilege-escalation surface.
    // ecs:DescribeTasks doesn't support resource-level scoping to a task
    // def, so it's scoped by cluster ARN via a condition instead.
    backendService.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ["ecs:RunTask"],
        resources: [mlTaskDefinition.taskDefinitionArn],
      }),
    );
    backendService.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ["ecs:DescribeTasks", "ecs:StopTask"],
        resources: ["*"],
        conditions: { ArnEquals: { "ecs:cluster": cluster.clusterArn } },
      }),
    );
    backendService.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: ["iam:PassRole"],
        resources: [
          mlTaskDefinition.taskRole.roleArn,
          mlTaskDefinition.executionRole!.roleArn,
        ],
      }),
    );
    bucket.grantReadWrite(backendService.taskDefinition.taskRole);

    // --- Frontend: Amplify Hosting, git-connected ---
    // The GitHub OAuth token must be created out-of-band (a fine-grained PAT
    // or the Amplify GitHub App install) and stored in Secrets Manager under
    // this name before first deploy - CDK can't provision the GitHub
    // connection itself.
    const amplifyApp = new amplify.App(this, "FrontendApp", {
      appName: resourceName(envConfig, "frontend"),
      sourceCodeProvider: new amplify.GitHubSourceCodeProvider({
        owner: "REPLACE_WITH_GITHUB_OWNER",
        repository: "REPLACE_WITH_GITHUB_REPO",
        oauthToken: SecretValue.secretsManager(resourceName(envConfig, "github-token")),
      }),
      buildSpec: codebuild.BuildSpec.fromObjectToYaml({
        version: 1,
        frontend: {
          phases: {
            preBuild: { commands: ["cd frontend", "npm ci"] },
            build: { commands: ["npm run build"] },
          },
          artifacts: {
            baseDirectory: "frontend/dist",
            files: ["**/*"],
          },
          cache: { paths: ["frontend/node_modules/**/*"] },
        },
      }),
      environmentVariables: {
        VITE_API_URL: `http://${backendService.loadBalancer.loadBalancerDnsName}`,
        VITE_COGNITO_USER_POOL_ID: userPool.userPoolId,
        VITE_COGNITO_CLIENT_ID: userPoolClient.userPoolClientId,
        VITE_AWS_REGION: this.region,
      },
    });
    amplifyApp.addBranch("main", { autoBuild: true, stage: "PRODUCTION" });

    new CfnOutput(this, "BackendAlbDnsName", { value: backendService.loadBalancer.loadBalancerDnsName });
    new CfnOutput(this, "AmplifyDefaultDomain", { value: amplifyApp.defaultDomain });
    new CfnOutput(this, "BackendRepoUri", { value: backendRepo.repositoryUri });
    new CfnOutput(this, "MlWorkerRepoUri", { value: mlWorkerRepo.repositoryUri });
    new CfnOutput(this, "MlTaskDefinitionArn", { value: mlTaskDefinition.taskDefinitionArn });
  }
}
