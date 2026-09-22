#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { FoundationStack } from "../lib/foundation-stack";
import { DataStack } from "../lib/data-stack";
import { AppStack } from "../lib/app-stack";
import { EnvConfig, resourceName } from "../config/constants";

const app = new cdk.App();

const envConfig: EnvConfig = {
  envName: app.node.tryGetContext("envName") ?? "prod",
  azCount: Number(app.node.tryGetContext("azCount") ?? 2),
};

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
};

const foundation = new FoundationStack(app, resourceName(envConfig, "foundation"), {
  env,
  envConfig,
});

const data = new DataStack(app, resourceName(envConfig, "data"), {
  env,
  envConfig,
  vpc: foundation.vpc,
  dbSg: foundation.dbSg,
});

new AppStack(app, resourceName(envConfig, "app"), {
  env,
  envConfig,
  vpc: foundation.vpc,
  dbSg: foundation.dbSg,
  dbInstance: data.dbInstance,
  bucket: data.bucket,
  userPool: data.userPool,
  userPoolClient: data.userPoolClient,
});
