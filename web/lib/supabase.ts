import { createClient } from "@supabase/supabase-js";
import { getLocalDayBounds } from "./date";
import { MOCK_PREDICTIONS, MOCK_RECORD, MOCK_SHAP } from "./mock";
import type { Prediction, RunningRecord, ShapValue } from "./types";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

/** null without web/.env.local: the build still succeeds and the pages show their empty states. */
export const supabase = supabaseUrl && supabaseKey ? createClient(supabaseUrl, supabaseKey) : null;

function db() {
  if (!supabase) {
    throw new Error(
      "Supabase is not configured: set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in web/.env.local"
    );
  }
  return supabase;
}

/**
 * `?demo=1` on any page serves the mock rows from lib/mock.ts instead of Supabase.
 * Dev server only: NODE_ENV is inlined at build time, so the static export ignores it.
 */
function isDemo(): boolean {
  return process.env.NODE_ENV !== "production"
    && typeof window !== "undefined"
    && new URLSearchParams(window.location.search).get("demo") === "1";
}

const HOUR_MS = 60 * 60 * 1000;

export async function getRecentPredictions(limit = 30): Promise<Prediction[]> {
  if (isDemo()) return MOCK_PREDICTIONS;
  const { data, error } = await db()
    .from("predictions")
    .select("*")
    .order("start_time_utc", { ascending: false })
    .limit(limit);

  if (error) throw error;
  return data ?? [];
}

export interface CardPrediction {
  /** all predictions of the anchor match's (local) day, by start time then match_id */
  predictions: Prediction[];
  /** position of the anchor match in `predictions` */
  index: number;
}

/**
 * For the /card widget: every prediction on the viewer's local day of the anchor
 * match, where the anchor is the next upcoming or live (unresolved, started less
 * than best_of × 1h ago) match, or if there is none, the most recently resolved one.
 */
export async function getCardPrediction(): Promise<CardPrediction | null> {
  if (isDemo()) {
    return { predictions: MOCK_PREDICTIONS, index: MOCK_PREDICTIONS.findIndex(p => p.actual_winner === null) };
  }
  const now = Date.now();
  const notOver = [1, 3, 5]
    .map(bo => `and(best_of.eq.${bo},start_time_utc.gte."${new Date(now - bo * HOUR_MS).toISOString()}")`)
    .join(",");

  const { data: upcoming, error: upcomingError } = await db()
    .from("predictions")
    .select("*")
    .is("actual_winner", null)
    .or(notOver)
    .order("start_time_utc", { ascending: true })
    .order("match_id", { ascending: true })
    .limit(1);

  if (upcomingError) throw upcomingError;
  let anchor: Prediction | undefined = upcoming?.[0];

  if (!anchor) {
    const { data: last, error: lastError } = await db()
      .from("predictions")
      .select("*")
      .not("actual_winner", "is", null)
      .order("start_time_utc", { ascending: false })
      .order("match_id", { ascending: false })
      .limit(1);

    if (lastError) throw lastError;
    anchor = last?.[0];
  }

  if (!anchor) return null;

  const [dayStart, dayEnd] = getLocalDayBounds(new Date(anchor.start_time_utc));
  const { data: day, error: dayError } = await db()
    .from("predictions")
    .select("*")
    .gte("start_time_utc", dayStart.toISOString())
    .lt("start_time_utc", dayEnd.toISOString())
    .order("start_time_utc", { ascending: true })
    .order("match_id", { ascending: true });

  if (dayError) throw dayError;
  const predictions = day && day.length > 0 ? day : [anchor];
  const index = Math.max(predictions.findIndex((p) => p.id === anchor.id), 0);
  return { predictions, index };
}

export async function getShapValues(predictionId: string): Promise<ShapValue[]> {
  if (isDemo()) return MOCK_SHAP[predictionId] ?? [];
  const { data, error } = await db()
    .from("shap_values")
    .select("*")
    .eq("prediction_id", predictionId)
    .order("shap_value", { ascending: false });

  if (error) throw error;
  return data ?? [];
}

export async function getRunningRecord(): Promise<RunningRecord | null> {
  if (isDemo()) return MOCK_RECORD;
  const { data, error } = await db()
    .from("running_record")
    .select("*")
    .eq("id", 1)
    .single();

  if (error) return null;
  return data;
}
