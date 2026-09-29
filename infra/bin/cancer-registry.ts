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
  region: process.env.CDK_DEFAULT_REGION ?? "us-west-2",
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

// The app stack needs a DNS name for the API's ACM certificate / HTTPS
// listener, so it is only synthesized once one is provided:
//   cdk deploy <foundation> <data>            (no domain needed; push images next)
//   cdk deploy -c apiDomainName=api-dev.example.edu <app>
const apiDomainName = app.node.tryGetContext("apiDomainName") as string | undefined;
if (!apiDomainName) {
  console.warn("apiDomainName context not set - skipping AppStack (pass -c apiDomainName=<fqdn>).");
} else {
  new AppStack(app, resourceName(envConfig, "app"), {
    env,
    envConfig,
    apiDomainName,
    deployFrontend: app.node.tryGetContext("deployFrontend") === "true",
    // Comma-separated, e.g. -c corsOrigins=https://main.d123.amplifyapp.com,http://localhost:5173
    corsOrigins: String(app.node.tryGetContext("corsOrigins") ?? "")
      .split(",")
      .map((o) => o.trim())
      .filter(Boolean),
    vpc: foundation.vpc,
    backendRepo: foundation.backendRepo,
    mlWorkerRepo: foundation.mlWorkerRepo,
    dbSg: foundation.dbSg,
    dbInstance: data.dbInstance,
    bucket: data.bucket,
    userPool: data.userPool,
    userPoolClient: data.userPoolClient,
  });
}
