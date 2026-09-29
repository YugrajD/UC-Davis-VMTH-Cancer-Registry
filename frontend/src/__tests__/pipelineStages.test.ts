import { describe, it, expect } from 'vitest';
import { STAGE_LABELS, LOCAL_STAGES, BATCH_STAGES } from '../components/shared/pipelineStages';

describe('STAGE_LABELS', () => {
  it('contains a label for every LOCAL_STAGES step', () => {
    for (const stage of LOCAL_STAGES) {
      expect(STAGE_LABELS).toHaveProperty(stage);
      expect(STAGE_LABELS[stage].length).toBeGreaterThan(0);
    }
  });

  it('contains a label for every BATCH_STAGES step', () => {
    for (const stage of BATCH_STAGES) {
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

describe('BATCH_STAGES', () => {
  it('starts with queued', () => {
    expect(BATCH_STAGES[0]).toBe('queued');
  });

  it('ends with ingesting', () => {
    expect(BATCH_STAGES[BATCH_STAGES.length - 1]).toBe('ingesting');
  });

  it('contains batch_running', () => {
    expect(BATCH_STAGES).toContain('batch_running');
  });

  it('has more stages than LOCAL_STAGES', () => {
    expect(BATCH_STAGES.length).toBeGreaterThan(LOCAL_STAGES.length);
  });

  it('submitting_batch_job comes before batch_running', () => {
    const submitIdx = BATCH_STAGES.indexOf('submitting_batch_job');
    const runIdx = BATCH_STAGES.indexOf('batch_running');
    expect(submitIdx).toBeLessThan(runIdx);
  });
});
