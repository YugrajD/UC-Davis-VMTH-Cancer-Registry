export const STAGE_LABELS: Record<string, string> = {
  queued:                  'Queued for processing',
  reading_files:           'Reading files',
  running_ml_worker:       'Running PetBERT inference',
  uploading_to_gcs:        'Uploading to Cloud Storage',
  submitting_batch_job:    'Submitting batch job',
  batch_queued:            'Batch job queued',
  batch_scheduled:         'Batch job scheduled',
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

export const GCP_STAGES = [
  'queued',
  'uploading_to_gcs',
  'submitting_batch_job',
  'batch_queued',
  'batch_scheduled',
  'batch_running',
  'downloading_predictions',
  'ingesting',
];
