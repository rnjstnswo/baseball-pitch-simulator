import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  title: string;
  probabilities: Record<string, number>;
  labels: Record<string, string>;
}

const BAR_COLOR = "#2a78d6"; // single measure → one hue, no legend
const INK_MUTED = "#898781";
const GRIDLINE = "#e1e0d9";

const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

export default function ProbabilityChart({ title, probabilities, labels }: Props) {
  const data = Object.entries(probabilities)
    .map(([key, p]) => ({ label: labels[key] ?? key, p }))
    .sort((a, b) => b.p - a.p);

  return (
    <div>
      <h3 className="mb-1 text-sm font-semibold text-gray-800">{title}</h3>
      <ResponsiveContainer width="100%" height={data.length * 34 + 30}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 48 }}>
          <CartesianGrid horizontal={false} stroke={GRIDLINE} />
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fill: INK_MUTED, fontSize: 11 }}
            axisLine={{ stroke: "#c3c2b7" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={100}
            tick={{ fill: "#52514e", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(value) => [pct(value as number), "Probability"]}
            cursor={{ fill: "rgba(11,11,11,0.04)" }}
          />
          <Bar dataKey="p" fill={BAR_COLOR} barSize={18} radius={[0, 4, 4, 0]}>
            <LabelList
              dataKey="p"
              position="right"
              formatter={(value) => pct(value as number)}
              style={{ fill: "#52514e", fontSize: 11 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
