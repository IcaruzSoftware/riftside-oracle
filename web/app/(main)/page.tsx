"use client";

import { useEffect, useState } from "react";
import { getRecentPredictions, getRunningRecord, getShapValues } from "@/lib/supabase";
import { getLocalDateString } from "@/lib/date";
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

      const { today, upcoming } = splitByDay(preds);

      // Fetch SHAP values for today's and upcoming matches (the ones shown as cards)
      const shapEntries = await Promise.all(
        [...today, ...upcoming].map(async (p): Promise<[string, ShapValue[]]> => {
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

  const {
    today: todaysPredictions,
    upcoming: upcomingPredictions,
    recent: recentPredictions,
  } = splitByDay(predictions);

  return (
    <div className="space-y-12">

      {/* Today's predictions */}
      <section>
        <SectionHeader icon="📅" label="TODAY'S PREDICTIONS" />
        {todaysPredictions.length === 0 ? (
          <EmptyState message={upcomingPredictions.length > 0
            ? "No matches scheduled today. See the upcoming matches below."
            : "No matches scheduled today. Check back tomorrow."} />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            {todaysPredictions.map(p => (
              <GameCard key={p.id} prediction={p} shap={shapMap[p.id]} />
            ))}
          </div>
        )}
      </section>

      {/* Upcoming predictions (later local days) */}
      {upcomingPredictions.length > 0 && (
        <section>
          <SectionHeader icon="⏭" label="UPCOMING" />
          <div className="grid gap-4 sm:grid-cols-2">
            {upcomingPredictions.map(p => (
              <GameCard key={p.id} prediction={p} shap={shapMap[p.id]} />
            ))}
          </div>
        </section>
      )}

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
              { label: "Series Accuracy", value: "70.2%", note: "6,260 held-out series, all leagues" },
              { label: "Major Leagues",   value: "69.9%", note: "1,453 held-out series" },
              { label: "Game Accuracy",   value: "65.3%", note: "single games, all leagues" },
              { label: "Log Loss",        value: "0.618", note: "per game; coin flip = 0.693" },
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
                Every pro player has a custom <span className="text-white">player ELO</span>: five
                skill ratings (combat, damage, economy, laning, vision), each compared only against
                players in the same role, plus an <span className="text-white">outcome ELO</span>{" "}
                that moves with wins and losses. Before a match, the two rosters&apos; ratings are
                combined with recent team form, head-to-head history, schedule and roster changes,
                and an <span className="text-white">XGBoost</span> classifier turns them into the
                chance of winning a single game, averaged over both sides of the map. That per-game
                chance is then compounded into the chance of winning the Bo1, Bo3 or Bo5 series.
              </p>
            </div>

            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-white mb-1">Training</p>
              <p>
                The model was trained on pro games from{" "}
                <span className="text-white">Oracle&apos;s Elixir</span> since 2017, split
                chronologically so it never sees the future. Hyperparameters were tuned with{" "}
                <span className="text-white">Optuna</span>, and the raw probabilities pass through a{" "}
                <span className="text-white">Platt scaler</span> so the percentages are well
                calibrated. The numbers above come from a held-out test period (May 2025 to
                October 2026) that was never used for any modelling choice.
              </p>
            </div>

            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-white mb-1">Limits</p>
              <p>
                Predictions are made <span className="text-white">before the draft</span>, so
                champion picks, which decide many games, are unknown. That puts a ceiling on any
                pre-draft model: about 65% per single game is close to it. Most of the signal
                comes from the outcome ELO alone (picking the higher-rated team is right in 69.8% of
                series), and many matchups are close to a coin flip.
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

/**
 * Groups by the viewer's local date of start_time_utc: today (ascending), unresolved
 * matches on later days (ascending), and earlier days (newest first).
 */
function splitByDay(preds: Prediction[]) {
  const today = getLocalDateString();
  const day = (p: Prediction) => getLocalDateString(new Date(p.start_time_utc));
  const byStart = (a: Prediction, b: Prediction) =>
    new Date(a.start_time_utc).getTime() - new Date(b.start_time_utc).getTime();
  return {
    today: preds.filter(p => day(p) === today).sort(byStart),
    upcoming: preds.filter(p => day(p) > today && p.actual_winner === null).sort(byStart),
    recent: preds.filter(p => day(p) < today).sort((a, b) => byStart(b, a)),
  };
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
