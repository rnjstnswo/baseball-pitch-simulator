import { useMemo, useState } from "react";

import { usePitchers } from "../api/hooks";
import type { PitcherSummary } from "../api/types";

interface Props {
  selected: PitcherSummary | null;
  onSelect: (pitcher: PitcherSummary) => void;
}

const MAX_RESULTS = 50;

export default function PitcherSelector({ selected, onSelect }: Props) {
  const { data, isPending, error } = usePitchers();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const matches = useMemo(() => {
    const pitchers = data?.pitchers ?? [];
    const q = query.trim().toLowerCase();
    const filtered = q
      ? pitchers.filter((p) => p.full_name.toLowerCase().includes(q))
      : pitchers;
    return filtered.slice(0, MAX_RESULTS);
  }, [data, query]);

  if (isPending) return <p className="text-sm text-gray-500">Loading pitchers…</p>;
  if (error)
    return <p className="text-sm text-red-600">Failed to load pitchers: {error.message}</p>;

  return (
    <div className="relative">
      <input
        type="text"
        placeholder="Search pitchers…"
        value={open ? query : (selected?.full_name ?? query)}
        onFocus={() => {
          setOpen(true);
          setQuery("");
        }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      {open && (
        <ul className="absolute z-10 mt-1 max-h-64 w-full overflow-auto rounded-md border border-gray-200 bg-white shadow-lg">
          {matches.length === 0 && (
            <li className="px-3 py-2 text-sm text-gray-500">No matches</li>
          )}
          {matches.map((p) => (
            <li key={p.pitcher_id}>
              <button
                type="button"
                // onMouseDown so the click wins the race against the input's onBlur
                onMouseDown={() => {
                  onSelect(p);
                  setOpen(false);
                }}
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-blue-50"
              >
                <span>{p.full_name}</span>
                <span className="text-xs text-gray-500">
                  {p.team} · {p.p_throws}HP
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
