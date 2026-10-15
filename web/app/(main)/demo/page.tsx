import GameCard from "@/components/GameCard";
import type { Prediction, ShapValue } from "@/lib/types";

const mockPrediction: Prediction = {
  id: "demo-1",
  game_id: "0000000000",
  game_date: new Date().toISOString().split("T")[0],
  game_time_utc: null,
  home_team: "NYK",
  away_team: "BOS",
  home_team_id: "1610612752",
  away_team_id: "1610612738",
  predicted_winner: "home",
  predicted_team: "NYK",
  home_win_prob: 0.629,
  away_win_prob: 0.371,
  confidence: 0.629,
  actual_winner: null,
  correct: null,
  created_at: new Date().toISOString(),
};

const mockShap: ShapValue[] = [
  { id: "s1", prediction_id: "demo-1", feature_name: "diff_net_last10",         shap_value:  0.1932, feature_value:  8.4 },
  { id: "s2", prediction_id: "demo-1", feature_name: "home_general_elo",         shap_value: -0.1094, feature_value: 24937 },
  { id: "s3", prediction_id: "demo-1", feature_name: "home_efficiency_elo",      shap_value: -0.1012, feature_value: 5897 },
  { id: "s4", prediction_id: "demo-1", feature_name: "net_efficiency_advantage", shap_value:  0.0977, feature_value:  6.1 },
  { id: "s5", prediction_id: "demo-1", feature_name: "elo_general_diff",         shap_value:  0.0602, feature_value:  1240 },
];

const mockResolved: Prediction = {
  ...mockPrediction,
  id: "demo-2",
  home_team: "NYK",
  away_team: "SAS",
  home_team_id: "1610612752",
  away_team_id: "1610612759",
  actual_winner: "NYK",
  correct: true,
  confidence: 0.642,
  home_win_prob: 0.642,
  away_win_prob: 0.358,
};

const mockResolvedShap: ShapValue[] = mockShap.map(s => ({ ...s, prediction_id: "demo-2" }));

export default function DemoPage() {
  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-bold text-white">Component Preview</h1>
        <span className="text-xs text-brand-muted bg-brand-card border border-brand-border px-2 py-0.5 rounded-full">
          demo / not real data
        </span>
      </div>

      <div>
        <p className="text-xs text-brand-muted tracking-widest uppercase mb-3">Pending game (with SHAP)</p>
        <div className="max-w-lg">
          <GameCard prediction={mockPrediction} shap={mockShap} />
        </div>
      </div>

      <div>
        <p className="text-xs text-brand-muted tracking-widest uppercase mb-3">Resolved game / correct</p>
        <div className="max-w-lg">
          <GameCard prediction={mockResolved} shap={mockResolvedShap} />
        </div>
      </div>

      <div>
        <p className="text-xs text-brand-muted tracking-widest uppercase mb-3">Resolved game / wrong</p>
        <div className="max-w-lg">
          <GameCard
            prediction={{ ...mockResolved, id: "demo-3", actual_winner: "SAS", correct: false }}
            shap={mockResolvedShap.map(s => ({ ...s, prediction_id: "demo-3" }))}
          />
        </div>
      </div>
    </div>
  );
}
