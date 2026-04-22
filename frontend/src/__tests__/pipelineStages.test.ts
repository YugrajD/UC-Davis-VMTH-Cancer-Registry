import { describe, it, expect } from 'vitest';
import { STAGE_LABELS, LOCAL_STAGES, GCP_STAGES } from '../components/shared/pipelineStages';

describe('STAGE_LABELS', () => {
  it('contains a label for every LOCAL_STAGES step', () => {
    for (const stage of LOCAL_STAGES) {
      expect(STAGE_LABELS).toHaveProperty(stage);
      expect(STAGE_LABELS[stage].length).toBeGreaterThan(0);
    }
  });

  it('contains a label for every GCP_STAGES step', () => {
    for (const stage of GCP_STAGES) {
      expect(STAGE_LABELS).toHaveProperty(stage);
      expect(STAGE_LABELS[stage].length).toBeGreaterThan(0);
    }
  });

  it('has human-readable labels (no underscores)', () => {
    for (const label of Object.values(STAGE_LABELS)) {
      expect(label).not.toContain('_');
    }
  });
});

describe('LOCAL_STAGES', () => {
  it('starts with queued', () => {
    expect(LOCAL_STAGES[0]).toBe('queued');
  });

  it('ends with ingesting', () => {
    expect(LOCAL_STAGES[LOCAL_STAGES.length - 1]).toBe('ingesting');
  });

  it('contains running_ml_worker', () => {
    expect(LOCAL_STAGES).toContain('running_ml_worker');
  });

  it('has at least 3 stages', () => {
    expect(LOCAL_STAGES.length).toBeGreaterThanOrEqual(3);
  });
});

describe('GCP_STAGES', () => {
  it('starts with queued', () => {
    expect(GCP_STAGES[0]).toBe('queued');
  });

  it('ends with ingesting', () => {
    expect(GCP_STAGES[GCP_STAGES.length - 1]).toBe('ingesting');
  });

  it('contains batch_running', () => {
    expect(GCP_STAGES).toContain('batch_running');
  });

  it('has more stages than LOCAL_STAGES', () => {
    expect(GCP_STAGES.length).toBeGreaterThan(LOCAL_STAGES.length);
  });

  it('uploading_to_gcs comes before batch_running', () => {
    const uploadIdx = GCP_STAGES.indexOf('uploading_to_gcs');
    const runIdx = GCP_STAGES.indexOf('batch_running');
    expect(uploadIdx).toBeLessThan(runIdx);
  });
});
