interface Option<T extends string | number> {
  value: T;
  label: string;
}

interface Props<T extends string | number> {
  options: Option<T>[];
  value: T;
  onChange: (value: T) => void;
}

/** Segmented single-select button row (counts, handedness, tiers…). */
export default function ButtonGroup<T extends string | number>({
  options,
  value,
  onChange,
}: Props<T>) {
  return (
    <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`px-3 py-1.5 text-sm border-r border-gray-300 last:border-r-0 ${
            opt.value === value
              ? "bg-blue-600 text-white font-medium"
              : "bg-white text-gray-700 hover:bg-gray-50"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
