"use client";

import { useEffect, useState } from "react";
import { getCardPrediction, getRunningRecord } from "@/lib/supabase";
import { teamName, type Prediction, type RunningRecord } from "@/lib/types";
import TeamLogo from "@/components/TeamLogo";

const HOUR_MS = 60 * 60 * 1000;

export default function CardPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [index, setIndex] = useState(0);
  const [record, setRecord] = useState<RunningRecord | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [card, rec] = await Promise.all([
          getCardPrediction(),
          getRunningRecord(),
        ]);
        if (cancelled) return;
        setPredictions(card?.predictions ?? []);
        setIndex(card?.index ?? 0);
        setRecord(rec);
      } catch {
        if (cancelled) return;
        setPredictions([]);
        setIndex(0);
        setRecord(null);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const prediction = predictions[index] ?? null;
  const isUpcoming = prediction ? prediction.actual_winner === null : true;
  const hasMany = predictions.length > 1;
  const correct = record?.total_correct ?? 0;
  const incorrect = record?.total_incorrect ?? 0;
  const total = correct + incorrect;
  const accuracy = total > 0 ? ((correct / total) * 100).toFixed(1) : "--";
  const confidence = prediction ? Math.round(prediction.confidence * 100) : null;
  // upcoming: the pick; resolved: the actual winner (the ✓/✗ line says whether the pick was right)
  const winnerName = prediction
    ? teamName(prediction, prediction.actual_winner ?? prediction.predicted_team)
    : null;

  const startMs = prediction ? new Date(prediction.start_time_utc).getTime() : null;
  const now = Date.now();
  const isLive = isUpcoming && prediction !== null && startMs !== null
    && startMs <= now && now < startMs + prediction.best_of * HOUR_MS;
  const startLabel = isUpcoming && startMs !== null && startMs > now
    ? new Date(startMs).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  return (
    <div className="relative w-full h-screen bg-brand-bg flex flex-col overflow-hidden">

      {/* Header */}
      <div className="px-5 py-3 border-b border-brand-border flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-brand-green text-lg">🔮</span>
          <span className="font-black tracking-tight text-white text-sm">
            LOL <span className="text-brand-green">PREDICTIONS</span>
          </span>
        </div>
        {isLive && (
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-green" />
            <span className="text-[10px] text-brand-green tracking-widest uppercase font-bold">LIVE</span>
          </div>
        )}
        {!isLive && startLabel && (
          <span className="text-[10px] text-brand-muted tracking-widest uppercase font-bold">
            {startLabel}
          </span>
        )}
      </div>

      {/* Game + Predicted winner */}
      <div className="relative flex-1 grid grid-cols-2 divide-x divide-brand-border border-b border-brand-border min-h-0">

        {/* Game */}
        <div className="px-5 py-3 flex flex-col">
          <p className="text-[9px] tracking-widest uppercase text-brand-muted mb-2 truncate">
            📅 {prediction
              ? `${prediction.league_name} · Bo${prediction.best_of}${hasMany ? ` · ${index + 1}/${predictions.length}` : ""}`
              : "NEXT MATCH"}
          </p>
          {prediction ? (
            <div key={prediction.id} className="flex items-center gap-3 flex-1">
              <div className="flex flex-col items-center gap-1">
                <TeamLogo src={prediction.team1_image} code={prediction.team1_code} size={40} />
                <span className="text-[10px] font-bold text-white tracking-wide uppercase">{prediction.team1_code}</span>
              </div>
              <span className={`text-[10px] font-semibold ${prediction.score ? "text-white" : "text-brand-muted"}`}>
                {prediction.score ?? "VS"}
              </span>
              <div className="flex flex-col items-center gap-1">
                <TeamLogo src={prediction.team2_image} code={prediction.team2_code} size={40} />
                <span className="text-[10px] font-bold text-white tracking-wide uppercase">{prediction.team2_code}</span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-brand-muted">No upcoming match</p>
          )}
        </div>

        {/* Predicted winner / result */}
        <div className="px-5 py-3 flex flex-col">
          <p className="text-[9px] tracking-widest uppercase text-brand-muted mb-2">
            🏆 {isUpcoming ? "PREDICTED WINNER" : "RESULT"}
          </p>
          {prediction ? (
            <div className="flex-1 flex flex-col justify-center min-w-0">
              <p className={`${winnerName && winnerName.length > 12 ? "text-sm" : "text-xl"} font-black text-white tracking-tight leading-none uppercase line-clamp-2 break-words`}>
                {winnerName}
              </p>
              {isUpcoming ? (
                <p className="text-xs text-brand-green font-semibold mt-1.5">
                  ({confidence}% Series Win)
                </p>
              ) : (
                <p className={`text-xs font-semibold mt-1.5 ${prediction.correct ? "text-brand-green" : "text-red-400"}`}>
                  {prediction.correct ? "✓ Correct" : "✗ Wrong"} · Pick {prediction.predicted_team} ({confidence}%)
                </p>
              )}
            </div>
          ) : (
            <p className="text-xs text-brand-muted">No prediction yet</p>
          )}
        </div>

        {hasMany && (
          <>
            <button
              type="button"
              aria-label="Previous match"
              disabled={index === 0}
              onClick={() => setIndex(index - 1)}
              className={`absolute left-0 top-1/2 !border-0 -translate-y-1/2 w-6 text-2xl leading-none text-white hover:text-brand-green ${index === 0 ? "invisible" : ""}`}
            >
              ‹
            </button>
            <button
              type="button"
              aria-label="Next match"
              disabled={index === predictions.length - 1}
              onClick={() => setIndex(index + 1)}
              className={`absolute right-0 top-1/2 !border-0 -translate-y-1/2 w-6 text-2xl leading-none text-white hover:text-brand-green ${index === predictions.length - 1 ? "invisible" : ""}`}
            >
              ›
            </button>
          </>
        )}
      </div>

      {/* Running record */}
      <div className="px-5 py-4 flex-shrink-0">
        <p className="text-[9px] tracking-widest uppercase text-brand-muted mb-3">
          📊 RUNNING RECORD
        </p>
        <div className="grid grid-cols-3 gap-2">
          <div>
            <div className="flex items-end gap-1">
              <span className="text-2xl font-black text-brand-green">{correct}</span>
              <span className="text-brand-green text-base mb-0.5">✓</span>
            </div>
            <p className="text-[9px] text-brand-muted tracking-widest">CORRECT</p>
          </div>
          <div>
            <div className="flex items-end gap-1">
              <span className="text-2xl font-black text-red-400">{incorrect}</span>
              <span className="text-red-400 text-base mb-0.5">✗</span>
            </div>
            <p className="text-[9px] text-brand-muted tracking-widest">INCORRECT</p>
          </div>
          <div>
            <div className="flex items-end gap-1">
              <span className="text-2xl font-black text-white">{accuracy}%</span>
              <span className="text-brand-green text-base mb-0.5">🎯</span>
            </div>
            <p className="text-[9px] text-brand-muted tracking-widest">ACCURACY</p>
          </div>
        </div>
      </div>

      {hasMany && (
        <div aria-hidden className="absolute bottom-1.5 left-1/2 -translate-x-1/2 flex gap-1">
          {predictions.map((p, i) => (
            <span
              key={p.id}
              className={`w-3 h-0.5 rounded-full ${i === index ? "bg-brand-green" : "bg-brand-border"}`}
            />
          ))}
        </div>
      )}

    </div>
  );
}
