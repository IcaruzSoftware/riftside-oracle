"use client";

import { useEffect, useState } from "react";
import { getRecentPredictions, getRunningRecord, getShapValues } from "@/lib/supabase";
import { getEasternDateString } from "@/lib/date";
import GameCard from "@/components/GameCard";
import RecordStats from "@/components/RecordStats";
import RecentPredictionsTable from "@/components/RecentPredictionsTable";
import type { Prediction, RunningRecord, ShapValue } from "@/lib/types";

export default function HomePage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [record, setRecord] = useState<RunningRecord | null>(null);
  const [shapMap, setShapMap] = useState<Record<string, ShapValue[]>>({});

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const [preds, rec] = await Promise.all([
        getRecentPredictions(50).catch(() => []),
        getRunningRecord().catch(() => null),
      ]);
      if (cancelled) return;
      setPredictions(preds);
      setRecord(rec);

      const today = getEasternDateString();
      const todaysPredictions = preds.filter(p => p.game_date === today);

      // Fetch SHAP values for today's games
      const shapEntries = await Promise.all(
        todaysPredictions.map(async (p): Promise<[string, ShapValue[]]> => {
          try {
            return [p.id, await getShapValues(p.id)];
          } catch {
            return [p.id, []];
          }
        })
      );
      if (cancelled) return;
      setShapMap(Object.fromEntries(shapEntries));
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const today = getEasternDateString();
  const todaysPredictions = predictions.filter(p => p.game_date === today);
  const recentPredictions = predictions.filter(p => p.game_date !== today);

  return (
    <div className="space-y-12">

      {/* Today's predictions */}
      <section>
        <SectionHeader icon="📅" label="TODAY'S PREDICTIONS" />
        {todaysPredictions.length === 0 ? (
          <EmptyState message="No games scheduled today. Check back tomorrow." />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            {todaysPredictions.map(p => (
              <GameCard key={p.id} prediction={p} shap={shapMap[p.id]} />
            ))}
          </div>
        )}
      </section>

      {/* Running record */}
      <section>
        <SectionHeader icon="📊" label="RUNNING RECORD" />
        <div className="bg-brand-card border border-brand-border rounded-2xl px-6 py-5">
          <RecordStats record={record} />
        </div>
      </section>

      {/* Recent predictions */}
      {recentPredictions.length > 0 && (
        <section>
          <SectionHeader icon="🕐" label="RECENT PREDICTIONS" />
          <RecentPredictionsTable predictions={recentPredictions} />
        </section>
      )}

      {/* About the model */}
      <section>
        <SectionHeader icon="🧠" label="ABOUT THE MODEL" />
        <div className="bg-brand-card border border-brand-border rounded-2xl px-6 py-6 space-y-6 text-sm text-brand-muted leading-relaxed">

          <div className="grid sm:grid-cols-4 gap-4">
            {[
              { label: "Accuracy",  value: "67.5%",  note: "on 2,116 held-out games" },
              { label: "AUC-ROC",   value: "0.729",  note: "higher = better separation" },
              { label: "Brier",     value: "0.209",  note: "lower = better calibration" },
              { label: "Log Loss",  value: "0.606",  note: "random guess = 0.693" },
            ].map(({ label, value, note }) => (
              <div key={label} className="bg-brand-bg rounded-xl px-4 py-3 border border-brand-border">
                <p className="text-[10px] tracking-widest uppercase text-brand-muted mb-1">{label}</p>
                <p className="text-xl font-black text-white">{value}</p>
                <p className="text-[10px] text-brand-muted mt-0.5">{note}</p>
              </div>
            ))}
          </div>

          <div className="space-y-4">
            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-white mb-1">How it works</p>
              <p>
                Each prediction is made by an <span className="text-white">XGBoost</span> binary
                classifier trained on <span className="text-white">14,108 NBA games</span> spanning
                11 seasons (2015-16 through 2025-26). Before every game, a feature vector is assembled
                from recent team form, head-to-head history, rest days, injury reports, and a custom{" "}
                <span className="text-white">player ELO system</span> that tracks 7 skill dimensions
                (scoring, efficiency, defense, playmaking, rebounding, and two composite ratings)
                across a player's entire career.
              </p>
            </div>

            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-white mb-1">Training</p>
              <p>
                The dataset was split chronologically: <span className="text-white">9,875 games</span> for
                training, 2,116 for calibration, and 2,116 for the held-out test set. This ensures
                the model never sees future games during training. Hyperparameters were tuned with{" "}
                <span className="text-white">Optuna</span> using time-series cross-validation.
                Raw probabilities are then passed through a{" "}
                <span className="text-white">Platt scaler</span> (logistic regression on the
                calibration set) to produce well-calibrated confidence values.
              </p>
            </div>

            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-white mb-1">Explainability</p>
              <p>
                Every prediction is accompanied by <span className="text-white">SHAP values</span>{" "}
                (SHapley Additive exPlanations) computed from the underlying tree model. The top 5
                factors shown on each card represent the features that moved the probability the most
                for that specific matchup. Green bars support the pick, red bars work against it.
              </p>
            </div>
          </div>

        </div>
      </section>

    </div>
  );
}

function SectionHeader({ icon, label }: { icon: string; label: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-brand-green">{icon}</span>
      <h2 className="text-xs font-bold tracking-[0.2em] text-brand-muted uppercase">
        {label}
      </h2>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="bg-brand-card border border-brand-border rounded-2xl px-6 py-10 text-center text-brand-muted text-sm">
      {message}
    </div>
  );
}
