import ButtonGroup from "./ButtonGroup";

export interface Situation {
  balls: number;
  strikes: number;
  outs: number;
  inning: number;
  score_diff: number;
  on_1b: boolean;
  on_2b: boolean;
  on_3b: boolean;
}

interface Props {
  situation: Situation;
  onChange: (update: Partial<Situation>) => void;
}

const range = (n: number) => Array.from({ length: n }, (_, i) => ({ value: i, label: String(i) }));

const RUNNERS: { key: "on_1b" | "on_2b" | "on_3b"; label: string }[] = [
  { key: "on_1b", label: "1B" },
  { key: "on_2b", label: "2B" },
  { key: "on_3b", label: "3B" },
];

export default function SituationPanel({ situation, onChange }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-4">
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-500">Balls</label>
          <ButtonGroup
            options={range(4)}
            value={situation.balls}
            onChange={(balls) => onChange({ balls })}
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-500">Strikes</label>
          <ButtonGroup
            options={range(3)}
            value={situation.strikes}
            onChange={(strikes) => onChange({ strikes })}
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-500">Outs</label>
          <ButtonGroup
            options={range(3)}
            value={situation.outs}
            onChange={(outs) => onChange({ outs })}
          />
        </div>
      </div>
      <div className="flex flex-wrap gap-4">
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-500">Inning</label>
          <select
            value={situation.inning}
            onChange={(e) => onChange({ inning: Number(e.target.value) })}
            className="rounded-md border border-gray-300 px-2 py-1.5 text-sm"
          >
            {Array.from({ length: 12 }, (_, i) => i + 1).map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-500">
            Score diff (batter − pitcher)
          </label>
          <input
            type="number"
            min={-10}
            max={10}
            value={situation.score_diff}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isInteger(n) && n >= -10 && n <= 10) onChange({ score_diff: n });
            }}
            className="w-20 rounded-md border border-gray-300 px-2 py-1.5 text-sm"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-gray-500">Runners on</label>
          <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
            {RUNNERS.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                onClick={() => onChange({ [key]: !situation[key] })}
                className={`px-3 py-1.5 text-sm border-r border-gray-300 last:border-r-0 ${
                  situation[key]
                    ? "bg-blue-600 text-white font-medium"
                    : "bg-white text-gray-700 hover:bg-gray-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
