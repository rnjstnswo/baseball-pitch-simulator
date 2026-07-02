import { useState } from "react";

import { usePredict } from "./api/hooks";
import type { BatterHand, BatterQualityTier, PitcherSummary } from "./api/types";
import BatterPanel from "./components/BatterPanel";
import PitcherSelector from "./components/PitcherSelector";
import PitchTypeSelector from "./components/PitchTypeSelector";
import ResultsPanel from "./components/ResultsPanel";
import SituationPanel, { type Situation } from "./components/SituationPanel";
import StrikeZone from "./components/StrikeZone";

const INITIAL_SITUATION: Situation = {
  balls: 0,
  strikes: 0,
  outs: 0,
  inning: 1,
  score_diff: 0,
  on_1b: false,
  on_2b: false,
  on_3b: false,
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-gray-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function App() {
  const [pitcher, setPitcher] = useState<PitcherSummary | null>(null);
  const [hand, setHand] = useState<BatterHand>("R");
  const [tier, setTier] = useState<BatterQualityTier>("average");
  const [situation, setSituation] = useState<Situation>(INITIAL_SITUATION);
  const [pitchType, setPitchType] = useState<string | null>(null);
  const [location, setLocation] = useState<{ plate_x: number; plate_z: number } | null>(
    null,
  );

  const predict = usePredict();

  function handleZoneClick(plate_x: number, plate_z: number) {
    if (!pitcher || !pitchType) return;
    setLocation({ plate_x, plate_z });
    predict.mutate({
      pitcher_id: pitcher.pitcher_id,
      pitch_type: pitchType,
      plate_x,
      plate_z,
      batter_hand: hand,
      batter_quality_tier: tier,
      ...situation,
    });
  }

  return (
    <div className="min-h-screen bg-[#f9f9f7] text-gray-900">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-lg font-bold">Baseball Pitch Simulator</h1>
        <p className="text-sm text-gray-500">
          What happens if this pitcher throws this pitch, here, right now?
        </p>
      </header>

      <main className="mx-auto grid max-w-7xl gap-4 p-4 lg:grid-cols-[minmax(320px,1fr)_minmax(300px,auto)_minmax(360px,1fr)]">
        <div className="space-y-4">
          <Section title="1 · Pitcher">
            <PitcherSelector
              selected={pitcher}
              onSelect={(p) => {
                setPitcher(p);
                setPitchType(null);
                setLocation(null);
                predict.reset();
              }}
            />
          </Section>
          <Section title="2 · Batter">
            <BatterPanel
              hand={hand}
              tier={tier}
              onHandChange={setHand}
              onTierChange={setTier}
            />
          </Section>
          <Section title="3 · Situation">
            <SituationPanel
              situation={situation}
              onChange={(update) => setSituation((s) => ({ ...s, ...update }))}
            />
          </Section>
          <Section title="4 · Pitch type">
            <PitchTypeSelector
              pitcherId={pitcher?.pitcher_id ?? null}
              selected={pitchType}
              onSelect={setPitchType}
            />
          </Section>
        </div>

        <Section title="5 · Location (catcher's view)">
          <StrikeZone
            location={location}
            onSelect={handleZoneClick}
            disabled={!pitcher || !pitchType}
          />
          {location && (
            <p className="mt-2 text-xs text-gray-500">
              plate_x {location.plate_x.toFixed(2)} ft · plate_z{" "}
              {location.plate_z.toFixed(2)} ft
            </p>
          )}
        </Section>

        <Section title="Prediction">
          <ResultsPanel
            result={predict.data ?? null}
            isPending={predict.isPending}
            error={predict.error}
          />
        </Section>
      </main>
    </div>
  );
}
