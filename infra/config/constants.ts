/** Shared naming/sizing constants referenced across stacks. */

export const APP_NAME = "cancer-registry";

export interface EnvConfig {
  envName: string;
  azCount: number;
}

export function resourceName(envConfig: EnvConfig, suffix: string): string {
  return `${APP_NAME}-${envConfig.envName}-${suffix}`;
}

export const BACKEND_CONTAINER_PORT = 8000;

// Backend is an always-on ApplicationLoadBalancedFargateService (min 1 task,
// no true scale-to-zero) - sized to match the current Cloud Run limits
// (0.5-1 vCPU / 256-512Mi), rounded up to the nearest valid Fargate combo.
export const BACKEND_TASK_CPU = 512; // 0.5 vCPU
export const BACKEND_TASK_MEMORY_MIB = 1024;
export const BACKEND_MIN_TASK_COUNT = 1;
export const BACKEND_MAX_TASK_COUNT = 10;
export const BACKEND_CPU_TARGET_UTILIZATION_PERCENT = 60;

// ML task is launched on-demand via ecs:RunTask (no Service, no idle cost) -
// sized to match the current GCP Batch machine type (n1-standard-4 = 4 vCPU /
// ~15GB, per backend/app/config.py's GCP_BATCH_MACHINE_TYPE default).
export const ML_TASK_CPU = 4096; // 4 vCPU
export const ML_TASK_MEMORY_MIB = 16384; // 16 GB

export const ECR_MAX_IMAGE_COUNT = 10;

export const S3_UPLOADS_PREFIX = "uploads/";
export const S3_REPORTS_PREFIX = "reports/";
export const S3_MODELS_PREFIX = "models/";

export const DB_NAME = "cancer_registry";
export const DB_PORT = 5432;
export const POSTGRES_ENGINE_MAJOR_VERSION = "16";
