import { useArsenal } from "../api/hooks";

interface Props {
  pitcherId: number | null;
  selected: string | null;
  onSelect: (pitchType: string) => void;
}

export default function PitchTypeSelector({ pitcherId, selected, onSelect }: Props) {
  const { data, isPending, error } = useArsenal(pitcherId);

  if (pitcherId === null)
    return <p className="text-sm text-gray-500">Select a pitcher first.</p>;
  if (isPending) return <p className="text-sm text-gray-500">Loading arsenal…</p>;
  if (error)
    return <p className="text-sm text-red-600">Failed to load arsenal: {error.message}</p>;

  return (
    <div className="flex flex-wrap gap-2">
      {data.arsenal.map((entry) => (
        <button
          key={entry.pitch_type}
          type="button"
          onClick={() => onSelect(entry.pitch_type)}
          className={`rounded-md border px-3 py-2 text-left text-sm ${
            entry.pitch_type === selected
              ? "border-blue-600 bg-blue-600 text-white"
              : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
          }`}
        >
          <span className="block font-medium">{entry.pitch_name}</span>
          <span
            className={`block text-xs ${
              entry.pitch_type === selected ? "text-blue-100" : "text-gray-500"
            }`}
          >
            {(entry.usage_pct * 100).toFixed(0)}% · {entry.avg_speed.toFixed(1)} mph
          </span>
        </button>
      ))}
    </div>
  );
}
