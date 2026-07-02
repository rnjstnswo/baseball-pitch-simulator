import type { PredictResponse, UpdatedState } from "../api/types";
import ProbabilityChart from "./ProbabilityChart";

interface Props {
  result: PredictResponse | null;
  isPending: boolean;
  error: Error | null;
}

const PITCH_OUTCOME_LABELS: Record<string, string> = {
  ball: "Ball",
  called_strike: "Called strike",
  swinging_strike: "Swinging strike",
  foul: "Foul",
  in_play: "In play",
  hit_by_pitch: "Hit by pitch",
};

const BIP_OUTCOME_LABELS: Record<string, string> = {
  out: "Out",
  single: "Single",
  double: "Double",
  triple: "Triple",
  home_run: "Home run",
};

function StateLine({ state }: { state: UpdatedState }) {
  const bases =
    [state.on_1b && "1B", state.on_2b && "2B", state.on_3b && "3B"]
      .filter(Boolean)
      .join(", ") || "empty";
  return (
    <p className="text-sm text-gray-700">
      Count <span className="font-medium">{state.balls}–{state.strikes}</span>
      {" · "}
      {state.outs} out{state.outs === 1 ? "" : "s"}
      {" · "}bases {bases}
      {state.at_bat_result && (
        <>
          {" · "}
          <span className="rounded bg-blue-50 px-1.5 py-0.5 text-xs font-medium text-blue-800">
            {state.at_bat_result.replace(/_/g, " ")}
          </span>
        </>
      )}
    </p>
  );
}

export default function ResultsPanel({ result, isPending, error }: Props) {
  if (isPending)
    return <p className="text-sm text-gray-500">Predicting…</p>;
  if (error)
    return <p className="text-sm text-red-600">Prediction failed: {error.message}</p>;
  if (!result)
    return (
      <p className="text-sm text-gray-500">
        Pick a pitcher and pitch type, then click a location in the strike zone.
      </p>
    );

  const predicted = result.pitch_outcome.prediction;

  return (
    <div className="space-y-5">
      <p className="text-sm text-gray-800">
        Most likely:{" "}
        <span className="font-semibold">
          {PITCH_OUTCOME_LABELS[predicted] ?? predicted}
        </span>{" "}
        ({(result.pitch_outcome.probabilities[predicted] * 100).toFixed(1)}%)
      </p>

      <ProbabilityChart
        title="Pitch outcome"
        probabilities={result.pitch_outcome.probabilities}
        labels={PITCH_OUTCOME_LABELS}
      />

      {result.bip_outcome && (
        <ProbabilityChart
          title="If put in play"
          probabilities={result.bip_outcome.probabilities}
          labels={BIP_OUTCOME_LABELS}
        />
      )}

      <div>
        <h3 className="mb-1 text-sm font-semibold text-gray-800">Why</h3>
        <p className="text-sm text-gray-700">{result.explanation}</p>
      </div>

      <div>
        <h3 className="mb-1 text-sm font-semibold text-gray-800">Usage context</h3>
        <p className="text-sm text-gray-700">
          This pitcher throws this pitch{" "}
          {(result.usage_context.pitch_usage_overall_pct * 100).toFixed(1)}% of the
          time overall, and{" "}
          {(result.usage_context.pitch_usage_in_count_pct * 100).toFixed(1)}% in{" "}
          {result.usage_context.count} counts ({result.usage_context.sample_size}{" "}
          pitches).
        </p>
      </div>

      <div>
        <h3 className="mb-1 text-sm font-semibold text-gray-800">
          Resulting state
        </h3>
        <StateLine state={result.updated_state} />
      </div>
    </div>
  );
}
