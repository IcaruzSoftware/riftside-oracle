"use client";

import { useEffect, useState } from "react";
import GameCard from "@/components/GameCard";
import { MOCK_PREDICTIONS, MOCK_SHAP } from "@/lib/mock";

const [wrong, correct, upcoming, noLogo] = MOCK_PREDICTIONS;

const examples = [
  { label: "Upcoming match (with SHAP)", prediction: upcoming },
  { label: "Resolved match / correct",   prediction: correct },
  { label: "Resolved match / wrong",     prediction: wrong },
  { label: "No logos / long team name",  prediction: noLogo },
];

export default function DemoPage() {
  // Mock dates are relative to the viewer's today: render the cards after mount so the
  // static export doesn't freeze them at build time.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-bold text-white">Component Preview</h1>
        <span className="text-xs text-brand-muted bg-brand-card border border-brand-border px-2 py-0.5 rounded-full">
          demo / not real data
        </span>
      </div>

      {mounted && examples.map(({ label, prediction }) => (
        <div key={prediction.id}>
          <p className="text-xs text-brand-muted tracking-widest uppercase mb-3">{label}</p>
          <div className="max-w-lg">
            <GameCard prediction={prediction} shap={MOCK_SHAP[prediction.id]} />
          </div>
        </div>
      ))}
    </div>
  );
}
