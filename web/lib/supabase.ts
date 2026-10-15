import { createClient } from "@supabase/supabase-js";
import { getEasternDateString } from "./date";
import type { Prediction, RunningRecord, ShapValue } from "./types";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseKey);

export async function getRecentPredictions(limit = 30): Promise<Prediction[]> {
  const { data, error } = await supabase
    .from("predictions")
    .select("*")
    .order("game_date", { ascending: false })
    .limit(limit);

  if (error) throw error;
  return data ?? [];
}

export interface CardPrediction {
  /** all predictions of the anchor game's day, by tip-off time then game_id */
  predictions: Prediction[];
  /** position of the anchor game in `predictions` */
  index: number;
}

/**
 * For the /card widget: every prediction on the day of the anchor game, where
 * the anchor is the next upcoming (unresolved) game, or if there is none, the
 * most recently resolved game.
 */
export async function getCardPrediction(): Promise<CardPrediction | null> {
  const cutoffIso = new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString();
  const todayEastern = getEasternDateString();

  const { data: upcoming, error: upcomingError } = await supabase
    .from("predictions")
    .select("*")
    .is("actual_winner", null)
    .or(
      `game_time_utc.gte."${cutoffIso}",and(game_time_utc.is.null,game_date.gte.${todayEastern})`
    )
    .order("game_time_utc", { ascending: true, nullsFirst: false })
    .order("game_date", { ascending: true })
    .order("game_id", { ascending: true })
    .limit(1);

  if (upcomingError) throw upcomingError;
  let anchor: Prediction | undefined = upcoming?.[0];

  if (!anchor) {
    const { data: last, error: lastError } = await supabase
      .from("predictions")
      .select("*")
      .not("actual_winner", "is", null)
      .order("game_date", { ascending: false })
      .order("game_time_utc", { ascending: false, nullsFirst: false })
      .order("game_id", { ascending: false })
      .limit(1);

    if (lastError) throw lastError;
    anchor = last?.[0];
  }

  if (!anchor) return null;

  const { data: day, error: dayError } = await supabase
    .from("predictions")
    .select("*")
    .eq("game_date", anchor.game_date)
    .order("game_time_utc", { ascending: true, nullsFirst: false })
    .order("game_id", { ascending: true });

  if (dayError) throw dayError;
  const predictions = day && day.length > 0 ? day : [anchor];
  const index = Math.max(predictions.findIndex((p) => p.id === anchor.id), 0);
  return { predictions, index };
}

export async function getShapValues(predictionId: string): Promise<ShapValue[]> {
  const { data, error } = await supabase
    .from("shap_values")
    .select("*")
    .eq("prediction_id", predictionId)
    .order("shap_value", { ascending: false });

  if (error) throw error;
  return data ?? [];
}

export async function getRunningRecord(): Promise<RunningRecord | null> {
  const { data, error } = await supabase
    .from("running_record")
    .select("*")
    .eq("id", 1)
    .single();

  if (error) return null;
  return data;
}
