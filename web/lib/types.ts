export interface Prediction {
  id: string;
  game_id: string;
  game_date: string;
  game_time_utc: string | null;
  home_team: string;
  away_team: string;
  home_team_id: string | null;
  away_team_id: string | null;
  predicted_winner: "home" | "away";
  predicted_team: string;
  home_win_prob: number | null;
  away_win_prob: number | null;
  confidence: number;
  actual_winner: string | null;
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
