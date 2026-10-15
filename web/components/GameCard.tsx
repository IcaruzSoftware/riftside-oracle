import type { Prediction, ShapValue } from "@/lib/types";
import { TEAM_NAMES } from "@/lib/teams";
import TeamLogo from "./TeamLogo";

const FEATURE_LABELS: Record<string, string> = {
  diff_net_last10:         "Net Rating Diff (L10)",
  diff_net_last5:          "Net Rating Diff (L5)",
  elo_general_diff:        "Overall ELO Edge",
  elo_efficiency_diff:     "Efficiency ELO Edge",
  elo_scoring_diff:        "Scoring ELO Edge",
  elo_defense_diff:        "Defense ELO Edge",
  elo_playmaking_diff:     "Playmaking ELO Edge",
  elo_rebounding_diff:     "Rebounding ELO Edge",
  net_efficiency_advantage:"Scoring Efficiency Edge",
  diff_win_pct_last10:     "Win % Diff (L10)",
  diff_win_pct_last5:      "Win % Diff (L5)",
  home_general_elo:        "Home Overall ELO",
  away_general_elo:        "Away Overall ELO",
  home_efficiency_elo:     "Home Efficiency ELO",
  away_efficiency_elo:     "Away Efficiency ELO",
  home_days_rest:          "Home Rest Days",
  away_days_rest:          "Away Rest Days",
  rest_diff:               "Rest Advantage",
  home_is_back_to_back:    "Home Back-to-Back",
  away_is_back_to_back:    "Away Back-to-Back",
  home_star_player_out:    "Home Star Out",
  away_star_player_out:    "Away Star Out",
  away_availability_score: "Away Availability",
  home_availability_score: "Home Availability",
  h2h_home_win_pct:        "Head-to-Head Win %",
};

function formatFeatureName(raw: string): string {
  return FEATURE_LABELS[raw] ?? raw.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

interface Props {
  prediction: Prediction;
  shap?: ShapValue[];
}

export default function GameCard({ prediction, shap }: Props) {
  const {
    home_team, away_team, home_team_id, away_team_id,
    predicted_team, home_win_prob, away_win_prob,
    confidence, actual_winner, correct, game_date,
  } = prediction;

  const homeProb = home_win_prob ?? 0.5;
  const awayProb = away_win_prob ?? (1 - homeProb);
  const homeName = TEAM_NAMES[home_team] ?? home_team;
  const awayName = TEAM_NAMES[away_team] ?? away_team;
  const date     = new Date(game_date + "T12:00:00").toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric",
  });

  const isPending = actual_winner === null;
  const topShap   = shap?.slice(0, 5) ?? [];
  const maxShap   = Math.max(...topShap.map(s => Math.abs(s.shap_value)), 0.01);

  return (
    <div className="bg-brand-card border border-brand-border rounded-2xl overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-brand-border">
        <span className="text-xs text-brand-muted tracking-widest uppercase">
          {date}
        </span>
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
        {/* Home - left */}
        <div className="flex flex-col items-center gap-2 flex-1">
          <TeamLogo abbrev={home_team} teamId={home_team_id} size={56} />
          <span className="text-sm font-bold text-white tracking-wide">{home_team}</span>
          <span className="text-xs text-brand-muted hidden sm:block">{homeName}</span>
        </div>

        {/* VS + probability bar */}
        <div className="flex flex-col items-center gap-3 flex-[2]">
          <span className="text-xs text-brand-muted tracking-[0.3em]">VS</span>

          {/* Prob bar - diverging from center, home on left */}
          <div className="w-full flex items-center h-2">
            {/* left half: home */}
            <div className="flex-1 flex justify-end h-full">
              {homeProb >= awayProb && (
                <div
                  className="h-full rounded-l-full bg-brand-green transition-all duration-500"
                  style={{ width: `${(homeProb - 0.5) * 200}%` }}
                />
              )}
            </div>
            {/* center dotted line */}
            <div className="w-px h-4 shrink-0" style={{ background: 'repeating-linear-gradient(to bottom, rgba(107,114,128,0.6) 0px, rgba(107,114,128,0.6) 2px, transparent 2px, transparent 4px)' }} />
            {/* right half: away */}
            <div className="flex-1 flex justify-start h-full">
              {awayProb > homeProb && (
                <div
                  className="h-full rounded-r-full bg-red-500 transition-all duration-500"
                  style={{ width: `${(awayProb - 0.5) * 200}%` }}
                />
              )}
            </div>
          </div>
          <div className="flex justify-between w-full text-xs text-brand-muted">
            <span>{(homeProb * 100).toFixed(0)}%</span>
            <span>{(awayProb * 100).toFixed(0)}%</span>
          </div>
        </div>

        {/* Away - right */}
        <div className="flex flex-col items-center gap-2 flex-1">
          <TeamLogo abbrev={away_team} teamId={away_team_id} size={56} />
          <span className="text-sm font-bold text-white tracking-wide">{away_team}</span>
          <span className="text-xs text-brand-muted hidden sm:block">{awayName}</span>
        </div>
      </div>

      {/* Prediction */}
      <div className="mx-5 mb-4 px-4 py-3 bg-brand-bg rounded-xl flex items-center justify-between">
        <div>
          <p className="text-[10px] tracking-widest text-brand-muted uppercase mb-0.5">
            Predicted Winner
          </p>
          <p className="text-lg font-black text-white tracking-wide">
            {TEAM_NAMES[predicted_team] ?? predicted_team}
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
              {TEAM_NAMES[actual_winner] ?? actual_winner}
            </p>
          </div>
        )}
      </div>

      {/* SHAP factors */}
      {topShap.length > 0 && (() => {
        const homeIsPredicted = predicted_team === home_team;
        return (
          <div className="px-5 pb-5">
            {/* headers match team layout: home on left, away on right */}
            <div className="flex items-center mb-2">
              <span className="text-[10px] tracking-widest text-brand-muted uppercase w-36 shrink-0">
                Top Factors
              </span>
              <div className="flex-1 flex text-[9px]">
                <span className={`flex-1 text-left pl-1 ${homeIsPredicted ? "text-brand-green" : "text-red-400"}`}>
                  {home_team} →
                </span>
                <span className={`flex-1 text-right pr-1 ${!homeIsPredicted ? "text-brand-green" : "text-red-400"}`}>
                  ← {away_team}
                </span>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              {topShap.map((s) => {
                // Direction matches team layout: left = home (shap > 0), right = away (shap < 0)
                const isHome     = s.shap_value > 0;
                // Color: green = supports the pick, red = works against it
                const favorsPick = isHome === homeIsPredicted;
                const pct        = (Math.abs(s.shap_value) / maxShap) * 50;
                return (
                  <div key={s.id} className="flex items-center gap-2 text-xs">
                    <span className="text-brand-muted w-36 shrink-0 truncate text-[11px]">
                      {formatFeatureName(s.feature_name)}
                    </span>

                    <div className="flex-1 flex items-center h-3">
                      {/* left = home side */}
                      <div className="flex-1 flex justify-end h-1.5">
                        {isHome && (
                          <div
                            className={`h-full rounded-l-full ${favorsPick ? "bg-brand-green" : "bg-red-500"}`}
                            style={{ width: `${pct * 2}%` }}
                          />
                        )}
                      </div>

                      {/* center dotted line */}
                      <div className="w-px h-4 shrink-0" style={{ background: 'repeating-linear-gradient(to bottom, rgba(107,114,128,0.6) 0px, rgba(107,114,128,0.6) 2px, transparent 2px, transparent 4px)' }} />

                      {/* right = away side */}
                      <div className="flex-1 flex justify-start h-1.5">
                        {!isHome && (
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
              Green = supports pick · Red = works against pick · Showing 5 of 30+ factors
            </p>
          </div>
        );
      })()}

    </div>
  );
}
