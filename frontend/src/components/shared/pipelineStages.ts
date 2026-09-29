export const STAGE_LABELS: Record<string, string> = {
  queued:                  'Queued for processing',
  reading_files:           'Reading files',
  running_ml_worker:       'Running PetBERT inference',
  submitting_batch_job:    'Starting ML task',
  batch_queued:            'ML task queued',
  batch_scheduled:         'ML task starting',
  batch_running:           'Running PetBERT inference',
  downloading_predictions: 'Downloading predictions',
  ingesting:               'Writing to database',
};

export const LOCAL_STAGES = [
  'queued',
  'reading_files',
  'running_ml_worker',
  'ingesting',
];

export const BATCH_STAGES = [
  'queued',
  'submitting_batch_job',
  'batch_queued',
  'batch_scheduled',
  'batch_running',
  'downloading_predictions',
  'ingesting',
];
