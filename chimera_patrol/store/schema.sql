CREATE TABLE IF NOT EXISTS cameras (
  id                TEXT PRIMARY KEY,
  name              TEXT NOT NULL,
  source_type       TEXT NOT NULL CHECK (source_type IN ('file','rtsp')),
  source            TEXT NOT NULL,                       -- 文件名或脱敏后的 RTSP 地址
  status            TEXT NOT NULL DEFAULT 'offline' CHECK (status IN ('online','offline')),
  status_changed_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sop_runs (
  id               TEXT PRIMARY KEY,                     -- 'sop_' + 12 位十六进制
  sop_id           TEXT NOT NULL,
  sop_name         TEXT NOT NULL,
  camera_id        TEXT NOT NULL REFERENCES cameras(id),
  status           TEXT NOT NULL CHECK (status IN ('running','completed','check_failed_retry')),
  operator         TEXT NOT NULL,
  triggered_ms     INTEGER NOT NULL,
  window_start_ms  INTEGER NOT NULL,
  window_end_ms    INTEGER NOT NULL,
  sample_count     INTEGER NOT NULL CHECK (sample_count >= 1),
  sample_times_json TEXT NOT NULL DEFAULT '[]',          -- JSON 数组：各采样时刻 ms
  failure_reason   TEXT,
  retry_of         TEXT REFERENCES sop_runs(id),
  created_ms       INTEGER NOT NULL,
  finished_ms      INTEGER,
  CHECK (window_end_ms >= window_start_ms),
  CHECK ((status = 'check_failed_retry') = (failure_reason IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_sop_runs_camera_created ON sop_runs(camera_id, created_ms DESC);
CREATE INDEX IF NOT EXISTS idx_sop_runs_status ON sop_runs(status);
-- 同一 Camera 同时只允许一个 running
CREATE UNIQUE INDEX IF NOT EXISTS uq_sop_runs_running_per_camera ON sop_runs(camera_id) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS sop_step_results (
  run_id      TEXT NOT NULL REFERENCES sop_runs(id) ON DELETE CASCADE,
  step_index  INTEGER NOT NULL CHECK (step_index >= 0),
  step_id     TEXT NOT NULL,
  description TEXT NOT NULL,
  status      TEXT NOT NULL CHECK (status IN ('pass','fail','unknown')),
  evidence    TEXT NOT NULL DEFAULT '',
  approximate INTEGER NOT NULL DEFAULT 0 CHECK (approximate IN (0,1)),
  PRIMARY KEY (run_id, step_id)
);

CREATE TABLE IF NOT EXISTS events (
  id                      TEXT PRIMARY KEY,              -- 'evt_' + 12 位十六进制
  camera_id               TEXT NOT NULL REFERENCES cameras(id),
  event_type              TEXT NOT NULL CHECK (event_type IN ('intrusion','loitering','smoke_fire','no_helmet','sop_violation')),
  detection_source        TEXT NOT NULL CHECK (detection_source IN ('yolo','vlm_fallback','sop')),
  status                  TEXT NOT NULL DEFAULT 'pending_review'
                          CHECK (status IN ('pending_review','confirmed','dismissed','needs_human','review_failed')),
  start_ms                INTEGER NOT NULL,
  end_ms                  INTEGER NOT NULL,
  cv_confidence           REAL CHECK (cv_confidence IS NULL OR (cv_confidence >= 0 AND cv_confidence <= 1)),
  track_id                INTEGER,
  roi_id                  TEXT,
  roi_type                TEXT CHECK (roi_type IS NULL OR roi_type IN ('intrusion','work_zone')),
  cv_detail_json          TEXT NOT NULL DEFAULT '{}',
  trajectory_json         TEXT NOT NULL DEFAULT '[]',    -- [[ts_ms,x1,y1,x2,y2],...]，≤ trajectory_max_points
  keyframe_annotated_path TEXT NOT NULL,                 -- 相对 evidence.base_dir
  keyframe_raw_path       TEXT NOT NULL,
  clip_path               TEXT,
  clip_status             TEXT NOT NULL DEFAULT 'pending' CHECK (clip_status IN ('pending','ready','failed')),
  vlm_verdict             TEXT CHECK (vlm_verdict IS NULL OR vlm_verdict IN ('true_positive','false_positive','uncertain')),
  vlm_confidence          REAL CHECK (vlm_confidence IS NULL OR (vlm_confidence >= 0 AND vlm_confidence <= 1)),
  vlm_description         TEXT,
  vlm_suggestion          TEXT,
  vlm_error               TEXT,
  vlm_reviewed_ms         INTEGER,
  review_skipped          INTEGER NOT NULL DEFAULT 0 CHECK (review_skipped IN (0,1)),
  sop_run_id              TEXT REFERENCES sop_runs(id),
  created_ms              INTEGER NOT NULL,
  updated_ms              INTEGER NOT NULL,
  CHECK (end_ms >= start_ms),
  CHECK ((event_type = 'sop_violation') = (sop_run_id IS NOT NULL)),
  CHECK ((event_type = 'sop_violation') = (detection_source = 'sop')),
  CHECK (clip_status <> 'ready' OR clip_path IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_events_created        ON events(created_ms DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_events_camera_created ON events(camera_id, created_ms DESC);
CREATE INDEX IF NOT EXISTS idx_events_type           ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_status         ON events(status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_events_sop_run  ON events(sop_run_id) WHERE sop_run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS event_status_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id    TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  from_status TEXT CHECK (from_status IS NULL OR from_status IN ('pending_review','confirmed','dismissed','needs_human','review_failed')),
  to_status   TEXT NOT NULL CHECK (to_status IN ('pending_review','confirmed','dismissed','needs_human','review_failed')),
  actor_type  TEXT NOT NULL CHECK (actor_type IN ('system','inspector')),
  actor       TEXT NOT NULL,           -- system：'vlm_reviewer'|'review_switch'|'sop_checker'|'startup_recovery'；inspector：operator 名
  reason      TEXT,
  created_ms  INTEGER NOT NULL,
  CHECK (from_status IS NOT NULL OR to_status = 'pending_review')
);
CREATE INDEX IF NOT EXISTS idx_status_log_event ON event_status_log(event_id, id);
