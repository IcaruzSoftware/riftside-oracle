import type { Prediction, RunningRecord, ShapValue } from "./types";

/**
 * Mock rows for /demo and for `?demo=1` on / and /card under `npm run dev` (see
 * lib/supabase.ts): a way to render the site without Supabase. Not real predictions.
 * Times are relative to the viewer's local today. Teams and logos are from the
 * 2026 Worlds schedule; SHG and TSW have no logo to show the fallback.
 */

const now = new Date();

function at(dayOffset: number, hour: number) {
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate() + dayOffset, hour);
  const iso = start.toISOString();
  return {
    start_time_utc: iso,
    match_date: iso.slice(0, 10),
    created_at: new Date(start.getTime() - 6 * 60 * 60 * 1000).toISOString(),
  };
}

const worlds = { league_slug: "worlds", league_name: "Worlds" } as const;
const playIns = { ...worlds, block_name: "Play-Ins", best_of: 5 } as const;

const team = {
  T1:  { code: "T1",  name: "T1",                            image: "http://static.lolesports.com/teams/1726801573959_539px-T1_2019_full_allmode.png" },
  KT:  { code: "KT",  name: "kt Rolster",                    image: "http://static.lolesports.com/teams/kt_darkbackground.png" },
  C9:  { code: "C9",  name: "Cloud9 Kia",                    image: "http://static.lolesports.com/teams/1736924120254_C9Kia_IconBlue_Transparent_2000x2000.png" },
  KC:  { code: "KC",  name: "Karmine Corp",                  image: "http://static.lolesports.com/teams/1704714951336_KC.png" },
  FUR: { code: "FUR", name: "FURIA",                         image: "http://static.lolesports.com/teams/FURIA---black.png" },
  MVK: { code: "MVK", name: "MVK Esports",                   image: "http://static.lolesports.com/teams/1767089709161_White_Logo.png" },
  SHG: { code: "SHG", name: "Fukuoka SoftBank HAWKS gaming", image: null },
  TSW: { code: "TSW", name: "Team Secret Whales",            image: null },
};

function teams(t1: keyof typeof team, t2: keyof typeof team) {
  const [a, b] = [team[t1], team[t2]];
  return {
    team1_code: a.code, team1_name: a.name, team1_image: a.image,
    team2_code: b.code, team2_name: b.name, team2_image: b.image,
  };
}

/** Ordered by start time: two resolved (past days), two today, one tomorrow. */
export const MOCK_PREDICTIONS: Prediction[] = [
  {
    id: "demo-1", match_id: "demo-match-1",
    ...worlds, block_name: "Swiss Stage", best_of: 3,
    ...at(-2, 12), ...teams("T1", "KT"),
    team1_game_prob: 0.58, team1_series_prob: 0.619, team2_series_prob: 0.381,
    predicted_team: "T1", confidence: 0.619,
    actual_winner: "KT", score: "1-2", correct: false,
  },
  {
    id: "demo-2", match_id: "demo-match-2",
    ...playIns, ...at(-1, 18), ...teams("C9", "KC"),
    team1_game_prob: 0.44, team1_series_prob: 0.3886, team2_series_prob: 0.6114,
    predicted_team: "KC", confidence: 0.6114,
    actual_winner: "KC", score: "1-3", correct: true,
  },
  {
    id: "demo-3", match_id: "demo-match-3",
    ...playIns, ...at(0, 14), ...teams("FUR", "MVK"),
    team1_game_prob: 0.47, team1_series_prob: 0.4439, team2_series_prob: 0.5561,
    predicted_team: "MVK", confidence: 0.5561,
    actual_winner: null, score: null, correct: null,
  },
  {
    id: "demo-4", match_id: "demo-match-4",
    ...playIns, ...at(0, 19), ...teams("SHG", "TSW"),
    team1_game_prob: 0.6, team1_series_prob: 0.6826, team2_series_prob: 0.3174,
    predicted_team: "SHG", confidence: 0.6826,
    actual_winner: null, score: null, correct: null,
  },
  {
    id: "demo-5", match_id: "demo-match-5",
    ...playIns, ...at(1, 16), ...teams("KC", "MVK"),
    team1_game_prob: 0.55, team1_series_prob: 0.5931, team2_series_prob: 0.4069,
    predicted_team: "KC", confidence: 0.5931,
    actual_winner: null, score: null, correct: null,
  },
];

const shapRows: [string, number, number][] = [
  ["diff_outcome_elo",             -0.2105, -48.2],
  ["diff_form_outcome_elo",        -0.0871, -12.5],
  ["diff_win_pct_last10",           0.0634,  0.2],
  ["diff_league_mean_outcome_elo", -0.0512, -61.0],
  ["diff_gd15_last10",              0.0298,  415.0],
];

export const MOCK_SHAP: Record<string, ShapValue[]> = Object.fromEntries(
  MOCK_PREDICTIONS.map(p => [
    p.id,
    shapRows.map(([feature_name, shap_value, feature_value], i) => ({
      id: `${p.id}-s${i + 1}`,
      prediction_id: p.id,
      feature_name,
      shap_value: p.predicted_team === p.team1_code ? -shap_value : shap_value,  // mostly toward the pick
      feature_value,
    })),
  ])
);

export const MOCK_RECORD: RunningRecord = {
  total_correct: 14,
  total_incorrect: 6,
  accuracy: 0.7,
  last_updated: now.toISOString(),
};
