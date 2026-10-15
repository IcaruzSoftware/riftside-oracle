-- Migration 001 — game_time_utc column + Row Level Security
-- Apply once in the Supabase SQL editor (idempotent; safe to re-run).

-- ---------------------------------------------------------------------------
-- 1. predictions.game_time_utc
-- Scheduled tip-off time (UTC) from the NBA schedule, written by predict.py.
-- ---------------------------------------------------------------------------
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS game_time_utc TIMESTAMPTZ;

-- ---------------------------------------------------------------------------
-- 2. Row Level Security — public SELECT only
--
-- The pipeline connects with the Supabase SERVICE-ROLE key, which bypasses RLS
-- entirely, so it keeps full read/write access. These policies govern only the
-- anon / public role used by the website: read-only, with no insert/update/delete
-- policy (and therefore no write access).
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
