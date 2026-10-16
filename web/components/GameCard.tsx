import { teamName, type Prediction, type ShapValue } from "@/lib/types";
import TeamLogo from "./TeamLogo";

// Model feature stems, i.e. without the blue_/red_/diff_ side prefix and the _last5/_last10 window.
// SHAP is computed with team1 as blue, so blue_ = team1, red_ = team2, diff_ = team1 - team2.
const FEATURE_LABELS: Record<string, string> = {
  outcome_elo:      "Outcome ELO",         general_elo:      "General ELO",
  combat_elo:       "Combat ELO",          damage_elo:       "Damage ELO",
  economy_elo:      "Economy ELO",         laning_elo:       "Laning ELO",
  vision_elo:       "Vision ELO",
  top_general_elo:  "Top General ELO",     top_outcome_elo:  "Top Outcome ELO",
  jng_general_elo:  "Jungle General ELO",  jng_outcome_elo:  "Jungle Outcome ELO",
  mid_general_elo:  "Mid General ELO",     mid_outcome_elo:  "Mid Outcome ELO",
  bot_general_elo:  "Bot General ELO",     bot_outcome_elo:  "Bot Outcome ELO",
  sup_general_elo:  "Support General ELO", sup_outcome_elo:  "Support Outcome ELO",
  form_outcome_elo: "Outcome ELO Form",    form_general_elo: "General ELO Form",
  win_pct:          "Win %",               gd15:             "Gold Diff @15",
  gamelength:       "Game Length",         dragons:          "Dragons",
  kills_pm:         "Kills / Min",         deaths_pm:        "Deaths / Min",
  firstblood_rate:  "First Blood %",       firstdragon_rate: "First Dragon %",
  firstherald_rate: "First Herald %",      firsttower_rate:  "First Tower %",
  firstbaron_rate:  "First Baron %",
  win_pct_on_blue:  "Win % on Blue",       win_pct_on_red:   "Win % on Red",
  win_pct_own_side: "Win % on Side",       streak:           "Streak",
  days_since_last:  "Days Since Last Game", games_last_7:    "Games in Last 7 Days",
  roster_churn:     "Shared Starters",     rookie_count:     "Rookies",
  league_mean_outcome_elo: "League Strength",
  h2h_games:        "Head-to-Head Games",  h2h_blue_wins:    "Head-to-Head Wins",
  h2h_blue_win_pct: "Head-to-Head Win %",
  is_international: "International Event", playoffs:        "Playoffs",
  league_tier:      "League Tier",
};

function formatFeatureName(raw: string, team1: string, team2: string): string {
  const [, side = "", rest = raw] = raw.match(/^(blue|red|diff)_(.+)$/) ?? [];
  const [, stem = rest, last = ""] = rest.match(/^(.+)_last(5|10)$/) ?? [];
  const label = FEATURE_LABELS[stem];
  if (!label) return raw;
  const full = last ? `${label} (L${last})` : label;
  if (side === "diff") return `${full} Diff`;
  return side ? `${side === "blue" ? team1 : team2} ${full}` : full;
}

interface Props {
  prediction: Prediction;
  shap?: ShapValue[];
}

export default function GameCard({ prediction, shap }: Props) {
  const {
    team1_code, team2_code, team1_name, team2_name, team1_image, team2_image,
    predicted_team, team1_series_prob: team1Prob, team2_series_prob: team2Prob,
    confidence, actual_winner, correct, score, start_time_utc, league_name, best_of,
  } = prediction;

  const date      = new Date(start_time_utc).toLocaleString("en-US", {
    month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
  });

  const isPending = actual_winner === null;
  const team1IsPredicted = predicted_team === team1_code;
  const topShap   = [...(shap ?? [])]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 5);
  const maxShap   = Math.max(...topShap.map(s => Math.abs(s.shap_value)), 0.01);

  return (
    <div className="bg-brand-card border border-brand-border rounded-2xl overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-brand-border">
        <div className="flex flex-col min-w-0">
          <span className="text-xs text-brand-muted tracking-widest uppercase truncate">
            {league_name} · Bo{best_of}
          </span>
          <span className="text-[10px] text-brand-muted">{date}</span>
        </div>
        {!isPending && (
          <span className={`text-xs font-bold tracking-widest px-2 py-0.5 rounded-full ${
            correct ? "bg-green-950 text-brand-green" : "bg-red-950 text-red-400"
          }`}>
            {correct ? "✓ CORRECT" : "✗ WRONG"}
          </span>
        )}
        {isPending && (
          <span className="text-xs text-brand-muted tracking-widest">PENDING</span>
        )}
      </div>

      {/* Teams */}
      <div className="flex items-center justify-between px-6 py-5 gap-4">
        {/* Team 1 - left */}
        <div className="flex flex-col items-center gap-2 flex-1">
          <TeamLogo src={team1_image} code={team1_code} size={56} />
          <span className="text-sm font-bold text-white tracking-wide">{team1_code}</span>
          <span className="text-xs text-brand-muted hidden sm:block text-center">{team1_name}</span>
        </div>

        {/* VS (or final score) + series probability bar */}
        <div className="flex flex-col items-center gap-3 flex-[2]">
          {score ? (
            <span className="text-sm font-bold text-white tracking-[0.3em]">{score.replace("-", " – ")}</span>
          ) : (
            <span className="text-xs text-brand-muted tracking-[0.3em]">VS</span>
          )}

          {/* Prob bar - split by series win %, team 1 on left; the pick's side green */}
          <div className="w-full flex h-2 gap-0.5 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${team1IsPredicted ? "bg-brand-green" : "bg-red-500"}`}
              style={{ width: `${team1Prob * 100}%` }}
            />
            <div
              className={`h-full transition-all duration-500 ${team1IsPredicted ? "bg-red-500" : "bg-brand-green"}`}
              style={{ width: `${team2Prob * 100}%` }}
            />
          </div>
          <div className="flex justify-between w-full text-xs text-brand-muted">
            <span>{(team1Prob * 100).toFixed(0)}%</span>
            <span className="text-[10px] tracking-widest uppercase">Series win</span>
            <span>{(team2Prob * 100).toFixed(0)}%</span>
          </div>
        </div>

        {/* Team 2 - right */}
        <div className="flex flex-col items-center gap-2 flex-1">
          <TeamLogo src={team2_image} code={team2_code} size={56} />
          <span className="text-sm font-bold text-white tracking-wide">{team2_code}</span>
          <span className="text-xs text-brand-muted hidden sm:block text-center">{team2_name}</span>
        </div>
      </div>

      {/* Prediction */}
      <div className="mx-5 mb-4 px-4 py-3 bg-brand-bg rounded-xl flex items-center justify-between">
        <div>
          <p className="text-[10px] tracking-widest text-brand-muted uppercase mb-0.5">
            Predicted Winner
          </p>
          <p className="text-lg font-black text-white tracking-wide">
            {teamName(prediction, predicted_team)}
            <span className="text-sm font-normal text-brand-muted ml-2">
              ({(confidence * 100).toFixed(0)}% confident)
            </span>
          </p>
        </div>
        {actual_winner && (
          <div className="text-right">
            <p className="text-[10px] tracking-widest text-brand-muted uppercase mb-0.5">
              Actual
            </p>
            <p className="text-lg font-black text-white">
              {teamName(prediction, actual_winner)}
            </p>
          </div>
        )}
      </div>

      {/* SHAP factors */}
      {topShap.length > 0 && (
        <div className="px-5 pb-5">
          {/* headers match team layout: team 1 on left, team 2 on right */}
          <div className="flex items-center mb-2">
            <span className="text-[10px] tracking-widest text-brand-muted uppercase w-36 shrink-0">
              Top Factors
            </span>
            <div className="flex-1 flex text-[9px]">
              <span className={`flex-1 text-left pl-1 ${team1IsPredicted ? "text-brand-green" : "text-red-400"}`}>
                {team1_code} →
              </span>
              <span className={`flex-1 text-right pr-1 ${!team1IsPredicted ? "text-brand-green" : "text-red-400"}`}>
                ← {team2_code}
              </span>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            {topShap.map((s) => {
              // Direction matches team layout: left = team 1 (shap > 0), right = team 2 (shap < 0)
              const isTeam1    = s.shap_value > 0;
              // Color: green = supports the pick, red = works against it
              const favorsPick = isTeam1 === team1IsPredicted;
              const pct        = (Math.abs(s.shap_value) / maxShap) * 50;
              return (
                <div key={s.id} className="flex items-center gap-2 text-xs">
                  <span className="text-brand-muted w-36 shrink-0 truncate text-[11px]">
                    {formatFeatureName(s.feature_name, team1_code, team2_code)}
                  </span>

                  <div className="flex-1 flex items-center h-3">
                    {/* left = team 1 side */}
                    <div className="flex-1 flex justify-end h-1.5">
                      {isTeam1 && (
                        <div
                          className={`h-full rounded-l-full ${favorsPick ? "bg-brand-green" : "bg-red-500"}`}
                          style={{ width: `${pct * 2}%` }}
                        />
                      )}
                    </div>

                    {/* center dotted line */}
                    <div className="w-px h-4 shrink-0" style={{ background: 'repeating-linear-gradient(to bottom, rgba(107,114,128,0.6) 0px, rgba(107,114,128,0.6) 2px, transparent 2px, transparent 4px)' }} />

                    {/* right = team 2 side */}
                    <div className="flex-1 flex justify-start h-1.5">
                      {!isTeam1 && (
                        <div
                          className={`h-full rounded-r-full ${favorsPick ? "bg-brand-green" : "bg-red-500"}`}
                          style={{ width: `${pct * 2}%` }}
                        />
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="text-[10px] text-brand-muted mt-3">
            Green = supports pick · Red = works against pick · Showing 5 of 136 factors
          </p>
        </div>
      )}

    </div>
  );
}
