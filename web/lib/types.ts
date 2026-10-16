export interface Prediction {
  id: string;
  match_id: string;
  league_slug: string;
  league_name: string;
  block_name: string | null;
  start_time_utc: string;
  match_date: string;
  best_of: 1 | 3 | 5;
  team1_code: string;
  team1_name: string;
  team1_image: string | null;
  team2_code: string;
  team2_name: string;
  team2_image: string | null;
  team1_game_prob: number;
  team1_series_prob: number;
  team2_series_prob: number;
  predicted_team: string;
  confidence: number;
  actual_winner: string | null;
  score: string | null;
  correct: boolean | null;
  created_at: string;
}

export interface ShapValue {
  id: string;
  prediction_id: string;
  feature_name: string;
  shap_value: number;
  feature_value: number | null;
}

export interface RunningRecord {
  total_correct: number;
  total_incorrect: number;
  accuracy: number | null;
  last_updated: string;
}

/** Full name of one of the row's two teams, by team code. */
export function teamName(p: Prediction, code: string): string {
  return code === p.team1_code ? p.team1_name : code === p.team2_code ? p.team2_name : code;
}
