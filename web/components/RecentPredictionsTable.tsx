"use client";

import { useState } from "react";
import GameCard from "@/components/GameCard";
import { getShapValues } from "@/lib/supabase";
import type { Prediction, ShapValue } from "@/lib/types";
import { TEAM_NAMES } from "@/lib/teams";

interface Props {
  predictions: Prediction[];
}

export default function RecentPredictionsTable({ predictions }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [shapCache, setShapCache]   = useState<Record<string, ShapValue[]>>({});
  const [loading, setLoading]       = useState<string | null>(null);

  async function toggle(id: string) {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    if (!shapCache[id]) {
      setLoading(id);
      try {
        const shap = await getShapValues(id);
        setShapCache(prev => ({ ...prev, [id]: shap }));
      } finally {
        setLoading(null);
      }
    }
  }

  return (
    <div className="bg-brand-card border border-brand-border rounded-2xl overflow-hidden">
      {/* Header row */}
      <div className="grid grid-cols-[1fr_1.5fr_1fr_1fr_40px] border-b border-brand-border text-brand-muted text-xs tracking-widest uppercase px-5 py-3">
        <span>Date</span>
        <span>Matchup</span>
        <span className="hidden sm:block">Predicted</span>
        <span className="text-right hidden sm:block">Confidence</span>
        <span />
      </div>

      {predictions.map((p) => {
        const isOpen = expandedId === p.id;
        const date = new Date(p.game_date + "T12:00:00").toLocaleDateString("en-US", {
          month: "short", day: "numeric",
        });

        return (
          <div key={p.id} className="border-b border-brand-border last:border-0">
            {/* Summary row */}
            <button
              onClick={() => toggle(p.id)}
              className="w-full grid grid-cols-[1fr_1.5fr_1fr_1fr_40px] items-center px-5 py-3 text-sm text-left hover:bg-white/[0.03] transition-colors"
            >
              <span className="text-brand-muted whitespace-nowrap">{date}</span>

              <span className="font-medium">
                {p.home_team} <span className="text-brand-muted">vs</span> {p.away_team}
              </span>

              <span className="text-brand-green hidden sm:block">
                {TEAM_NAMES[p.predicted_team] ?? p.predicted_team}
              </span>

              <span className="text-right text-brand-muted hidden sm:block">
                {(p.confidence * 100).toFixed(0)}%
                {p.correct === null ? "" : p.correct
                  ? <span className="ml-2 text-brand-green">✓</span>
                  : <span className="ml-2 text-red-400">✗</span>
                }
              </span>

              {/* Chevron */}
              <span className={`text-brand-muted text-xs transition-transform duration-200 text-right ${isOpen ? "rotate-180" : ""}`}>
                ▾
              </span>
            </button>

            {/* Expanded GameCard */}
            {isOpen && (
              <div className="px-4 pb-4">
                {loading === p.id ? (
                  <div className="text-brand-muted text-xs py-4 text-center">Loading…</div>
                ) : (
                  <GameCard prediction={p} shap={shapCache[p.id] ?? []} />
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
