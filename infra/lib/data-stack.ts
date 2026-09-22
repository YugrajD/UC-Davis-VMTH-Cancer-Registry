import { CustomResource, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as rds from "aws-cdk-lib/aws-rds";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { PythonFunction } from "@aws-cdk/aws-lambda-python-alpha";
import * as cr from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";
import * as path from "path";
import {
  DB_NAME,
  DB_PORT,
  EnvConfig,
  resourceName,
} from "../config/constants";

export interface DataStackProps extends StackProps {
  envConfig: EnvConfig;
  vpc: ec2.IVpc;
  dbSg: ec2.ISecurityGroup;
}

export class DataStack extends Stack {
  public readonly dbInstance: rds.DatabaseInstance;
  public readonly bucket: s3.Bucket;
  public readonly userPool: cognito.UserPool;
  public readonly userPoolClient: cognito.UserPoolClient;

  constructor(scope: Construct, id: string, props: DataStackProps) {
    super(scope, id, props);

    const { envConfig, vpc, dbSg } = props;

    this.dbInstance = new rds.DatabaseInstance(this, "Database", {
      instanceIdentifier: resourceName(envConfig, "db"),
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.VER_16,
      }),
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      securityGroups: [dbSg],
      databaseName: DB_NAME,
      port: DB_PORT,
      credentials: rds.Credentials.fromGeneratedSecret("postgres", {
        secretName: resourceName(envConfig, "db-credentials"),
      }),
      multiAz: false,
      storageEncrypted: true,
      allocatedStorage: 20,
      maxAllocatedStorage: 100,
      backupRetention: Duration.days(7),
      deletionProtection: envConfig.envName === "prod",
      removalPolicy: envConfig.envName === "prod" ? RemovalPolicy.SNAPSHOT : RemovalPolicy.DESTROY,
    });

    // RDS doesn't enable PostGIS by default - CREATE EXTENSION must run once
    // against the new database. This Lambda runs in the VPC (the DB has no
    // public endpoint) and is invoked via a Custom Resource on every
    // deploy/update; CREATE EXTENSION IF NOT EXISTS is idempotent so re-runs
    // are safe. PythonFunction bundles psycopg2-binary at synth time via
    // Docker, so no manually-built Lambda layer is needed.
    // Explicit SG (rather than the Lambda's auto-created default) so the
    // ingress rule granting it access to dbSg can be added as a standalone
    // resource below, without dbSg.addIngressRule() making FoundationStack
    // (which owns dbSg) depend on this stack's Lambda SG - the reverse of
    // the dependency that already exists (DataStack depends on Foundation
    // for the VPC/dbSg), which would be a cycle.
    const postgisBootstrapFnSg = new ec2.SecurityGroup(this, "PostgisBootstrapFnSg", {
      securityGroupName: resourceName(envConfig, "postgis-bootstrap-sg"),
      vpc,
      description: "PostGIS bootstrap Lambda (Custom Resource)",
      allowAllOutbound: true,
    });

    const postgisBootstrapFn = new PythonFunction(this, "PostgisBootstrapFn", {
      functionName: resourceName(envConfig, "postgis-bootstrap"),
      entry: path.join(__dirname, "..", "lambda", "postgis-bootstrap"),
      runtime: lambda.Runtime.PYTHON_3_12,
      index: "index.py",
      handler: "handler",
      timeout: Duration.seconds(30),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      securityGroups: [postgisBootstrapFnSg],
    });

    new ec2.CfnSecurityGroupIngress(this, "DbIngressFromPostgisBootstrap", {
      groupId: dbSg.securityGroupId,
      sourceSecurityGroupId: postgisBootstrapFnSg.securityGroupId,
      ipProtocol: "tcp",
      fromPort: DB_PORT,
      toPort: DB_PORT,
      description: "PostGIS bootstrap Lambda runs CREATE EXTENSION once per deploy",
    });
    this.dbInstance.secret!.grantRead(postgisBootstrapFn);

    new CustomResource(this, "PostgisBootstrap", {
      serviceToken: new cr.Provider(this, "PostgisBootstrapProvider", {
        onEventHandler: postgisBootstrapFn,
      }).serviceToken,
      properties: {
        Host: this.dbInstance.dbInstanceEndpointAddress,
        Port: this.dbInstance.dbInstanceEndpointPort,
        DbName: DB_NAME,
        User: this.dbInstance.secret!.secretValueFromJson("username").unsafeUnwrap(),
        Password: this.dbInstance.secret!.secretValueFromJson("password").unsafeUnwrap(),
      },
    });

    this.bucket = new s3.Bucket(this, "Bucket", {
      bucketName: resourceName(envConfig, "storage").toLowerCase(),
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: false,
      removalPolicy: envConfig.envName === "prod" ? RemovalPolicy.RETAIN : RemovalPolicy.DESTROY,
      lifecycleRules: [
        {
          id: "abort-incomplete-multipart-uploads",
          abortIncompleteMultipartUploadAfter: Duration.days(7),
        },
      ],
    });

    this.userPool = new cognito.UserPool(this, "UserPool", {
      userPoolName: resourceName(envConfig, "users"),
      selfSignUpEnabled: true,
      signInAliases: { email: true },
      autoVerify: { email: true },
      standardAttributes: {
        email: { required: true, mutable: false },
      },
      passwordPolicy: {
        minLength: 12,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
      },
      accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
      removalPolicy: envConfig.envName === "prod" ? RemovalPolicy.RETAIN : RemovalPolicy.DESTROY,
    });

    this.userPoolClient = this.userPool.addClient("WebClient", {
      userPoolClientName: resourceName(envConfig, "web-client"),
      authFlows: { userSrp: true },
      generateSecret: false,
      oAuth: {
        flows: { authorizationCodeGrant: true },
        scopes: [cognito.OAuthScope.EMAIL, cognito.OAuthScope.OPENID, cognito.OAuthScope.PROFILE],
      },
    });

    // Google OAuth federation requires a client ID/secret from the Google
    // Cloud console - out of CDK's reach to provision, so it's left as a
    // manual follow-up (add a UserPoolIdentityProviderGoogle construct here
    // once those credentials exist, then add "Google" to the client's
    // supportedIdentityProviders).
  }
}
