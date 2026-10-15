-- Courtside Oracle — Supabase schema
-- Run this once in the Supabase SQL editor to set up all tables.
-- Enable Row Level Security per table as appropriate for your access patterns.

-- ---------------------------------------------------------------------------
-- predictions
-- One row per game per day. "correct" is NULL until evaluate.py runs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id          TEXT NOT NULL UNIQUE,
    game_date        DATE NOT NULL,
    game_time_utc    TIMESTAMPTZ,     -- scheduled tip-off (UTC), from the NBA schedule
    home_team        TEXT NOT NULL,   -- team abbreviation, e.g. "LAL"
    away_team        TEXT NOT NULL,
    home_team_id     TEXT,            -- NBA team ID for logo URL
    away_team_id     TEXT,
    predicted_winner TEXT NOT NULL CHECK (predicted_winner IN ('home', 'away')),
    predicted_team   TEXT NOT NULL,   -- abbreviation of predicted winning team
    home_win_prob    NUMERIC(5, 4),   -- e.g. 0.6290
    away_win_prob    NUMERIC(5, 4),
    confidence       NUMERIC(5, 4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    actual_winner    TEXT,            -- abbreviation, NULL until game is played
    correct          BOOLEAN,         -- NULL until result is known
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_game_date ON predictions (game_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_correct    ON predictions (correct);

-- ---------------------------------------------------------------------------
-- shap_values
-- One row per feature per prediction. Drives the per-game waterfall charts.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shap_values (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id  UUID NOT NULL REFERENCES predictions (id) ON DELETE CASCADE,
    feature_name   TEXT NOT NULL,     -- human-readable, e.g. "Home Team Rest Days"
    shap_value     NUMERIC NOT NULL,  -- positive = pushes toward home win
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
-- model_metadata
-- One row per trained model version. Append-only — never overwrite.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_metadata (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version       TEXT NOT NULL,
    trained_on_date     DATE NOT NULL,
    seasons_covered     TEXT,             -- e.g. "2000-01 through 2023-24"
    accuracy_on_test    NUMERIC(5, 4),
    auc_roc             NUMERIC(5, 4),
    brier_score         NUMERIC(5, 4),
    feature_list        JSONB,            -- array of feature name strings
    best_params         JSONB,            -- XGBoost hyperparameters used
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Row Level Security — public SELECT only
--
-- The pipeline connects with the SERVICE-ROLE key, which bypasses RLS entirely
-- (full read/write). These policies govern only the anon/public role used by the
-- website: read-only, with no write policy (so no anon writes). See also
-- migrations/001_game_time_and_rls.sql.
-- ---------------------------------------------------------------------------
ALTER TABLE predictions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE shap_values    ENABLE ROW LEVEL SECURITY;
ALTER TABLE running_record ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metadata ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "public read predictions"    ON predictions;
DROP POLICY IF EXISTS "public read shap_values"    ON shap_values;
DROP POLICY IF EXISTS "public read running_record" ON running_record;
DROP POLICY IF EXISTS "public read model_metadata" ON model_metadata;

CREATE POLICY "public read predictions"    ON predictions    FOR SELECT USING (true);
CREATE POLICY "public read shap_values"    ON shap_values    FOR SELECT USING (true);
CREATE POLICY "public read running_record" ON running_record FOR SELECT USING (true);
CREATE POLICY "public read model_metadata" ON model_metadata FOR SELECT USING (true);
