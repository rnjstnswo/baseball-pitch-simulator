import type { BatterHand, BatterQualityTier } from "../api/types";
import ButtonGroup from "./ButtonGroup";

interface Props {
  hand: BatterHand;
  tier: BatterQualityTier;
  onHandChange: (hand: BatterHand) => void;
  onTierChange: (tier: BatterQualityTier) => void;
}

const TIERS: { value: BatterQualityTier; label: string }[] = [
  { value: "below_avg", label: "Below avg" },
  { value: "average", label: "Average" },
  { value: "above_avg", label: "Above avg" },
  { value: "elite", label: "Elite" },
];

export default function BatterPanel({ hand, tier, onHandChange, onTierChange }: Props) {
  return (
    <div className="space-y-3">
      <div>
        <label className="mb-1 block text-xs font-medium text-gray-500">Handedness</label>
        <ButtonGroup
          options={[
            { value: "L", label: "Left" },
            { value: "R", label: "Right" },
          ]}
          value={hand}
          onChange={onHandChange}
        />
      </div>
      <div>
        <label className="mb-1 block text-xs font-medium text-gray-500">
          Quality tier (wOBA)
        </label>
        <ButtonGroup options={TIERS} value={tier} onChange={onTierChange} />
      </div>
    </div>
  );
}
