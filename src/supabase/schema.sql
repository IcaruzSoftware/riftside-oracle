-- Riftside Oracle — Supabase schema
-- Run this once in the Supabase SQL editor to set up all tables.
-- Enable Row Level Security per table as appropriate for your access patterns.

-- ---------------------------------------------------------------------------
-- predictions
-- One row per lolesports match (series). actual_winner/score/correct are NULL
-- until evaluate.py resolves the match.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id           TEXT NOT NULL UNIQUE,      -- lolesports match id
    league_slug        TEXT NOT NULL,             -- e.g. "lck"
    league_name        TEXT NOT NULL,
    block_name         TEXT,                      -- e.g. "Playoffs", "Week 3"
    start_time_utc     TIMESTAMPTZ NOT NULL,
    match_date         DATE NOT NULL,             -- UTC date of start_time_utc
    best_of            SMALLINT NOT NULL CHECK (best_of IN (1, 3, 5)),
    team1_code TEXT NOT NULL, team1_name TEXT NOT NULL, team1_image TEXT,
    team2_code TEXT NOT NULL, team2_name TEXT NOT NULL, team2_image TEXT,
    team1_game_prob    NUMERIC(5, 4) NOT NULL,    -- side-neutral per-game p
    team1_series_prob  NUMERIC(5, 4) NOT NULL,
    team2_series_prob  NUMERIC(5, 4) NOT NULL,
    predicted_team     TEXT NOT NULL,             -- team code
    confidence         NUMERIC(5, 4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    actual_winner      TEXT,                      -- team code, NULL until resolved
    score              TEXT,                      -- team1-team2 game wins, e.g. "3-1"
    correct            BOOLEAN,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_match_date ON predictions (match_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_correct    ON predictions (correct);

-- ---------------------------------------------------------------------------
-- shap_values
-- One row per feature per prediction. Drives the per-game waterfall charts.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shap_values (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id  UUID NOT NULL REFERENCES predictions (id) ON DELETE CASCADE,
    feature_name   TEXT NOT NULL,     -- model feature, e.g. "diff_outcome_elo"
    shap_value     NUMERIC NOT NULL,  -- positive = pushes toward team1
    feature_value  NUMERIC           -- raw feature value for display context
);

CREATE INDEX IF NOT EXISTS idx_shap_prediction_id ON shap_values (prediction_id);

-- ---------------------------------------------------------------------------
-- running_record
-- Single-row table. Upserted by evaluate.py after each daily run.
-- Exposed to the website ticker widget.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS running_record (
    id               INTEGER PRIMARY KEY DEFAULT 1,  -- enforces single-row
    total_correct    INTEGER NOT NULL DEFAULT 0,
    total_incorrect  INTEGER NOT NULL DEFAULT 0,
    accuracy         NUMERIC(5, 4),                  -- recomputed on every update
    last_updated     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed with zeroes so the ticker always has a row to read
INSERT INTO running_record (id, total_correct, total_incorrect, accuracy)
VALUES (1, 0, 0, NULL)
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Row Level Security — public SELECT only
--
-- The pipeline connects with the SERVICE-ROLE key, which bypasses RLS entirely
-- (full read/write). These policies govern only the anon/public role used by the
-- website: read-only, with no write policy (so no anon writes).
-- ---------------------------------------------------------------------------
ALTER TABLE predictions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE shap_values    ENABLE ROW LEVEL SECURITY;
ALTER TABLE running_record ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "public read predictions"    ON predictions;
DROP POLICY IF EXISTS "public read shap_values"    ON shap_values;
DROP POLICY IF EXISTS "public read running_record" ON running_record;

CREATE POLICY "public read predictions"    ON predictions    FOR SELECT USING (true);
CREATE POLICY "public read shap_values"    ON shap_values    FOR SELECT USING (true);
CREATE POLICY "public read running_record" ON running_record FOR SELECT USING (true);
