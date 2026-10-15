import type { RunningRecord } from "@/lib/types";

interface Props {
  record: RunningRecord | null;
}

export default function RecordStats({ record }: Props) {
  const correct   = record?.total_correct   ?? 0;
  const incorrect = record?.total_incorrect ?? 0;
  const accuracy  = record?.accuracy != null
    ? (record.accuracy * 100).toFixed(1)
    : correct + incorrect > 0
      ? ((correct / (correct + incorrect)) * 100).toFixed(1)
      : "-";

  return (
    <div className="flex items-center gap-6 flex-wrap">
      <Stat value={correct}   label="CORRECT"   color="text-brand-green" icon="✓" />
      <Stat value={incorrect} label="INCORRECT"  color="text-red-500"    icon="✗" />
      <Stat value={`${accuracy}%`} label="ACCURACY" color="text-white" icon="◎" />
    </div>
  );
}

function Stat({
  value,
  label,
  color,
  icon,
}: {
  value: number | string;
  label: string;
  color: string;
  icon: string;
}) {
  return (
    <div className="flex flex-col items-center gap-1 min-w-[90px]">
      <div className={`flex items-center gap-2 text-4xl font-bold ${color}`}>
        <span>{value}</span>
        <span className="text-2xl">{icon}</span>
      </div>
      <span className="text-xs tracking-widest text-brand-muted">{label}</span>
    </div>
  );
}
