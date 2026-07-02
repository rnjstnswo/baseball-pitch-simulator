// TypeScript mirrors of the Pydantic schemas in api/schemas.py.
// The /predict contract is frozen (docs/PROJECT_SPEC.md §6) — do not change
// field names or types without updating the spec and the API version.

export type BatterQualityTier = "below_avg" | "average" | "above_avg" | "elite";

export type BatterHand = "L" | "R";

export interface PredictRequest {
  pitcher_id: number;
  pitch_type: string;
  plate_x: number; // feet from center, -2.0 to 2.0
  plate_z: number; // feet from ground, 0.5 to 5.0
  batter_hand: BatterHand;
  batter_quality_tier: BatterQualityTier;
  balls: number; // 0-3
  strikes: number; // 0-2
  outs: number; // 0-2
  inning: number; // 1-12
  score_diff: number; // batter team score − pitcher team score, -10 to 10
  on_1b: boolean;
  on_2b: boolean;
  on_3b: boolean;
}

export interface PitchOutcomeResult {
  prediction: string;
  probabilities: Record<string, number>;
}

export interface BIPOutcomeResult {
  prediction: string;
  probabilities: Record<string, number>;
}

export interface ShapFactor {
  feature: string;
  value: number | string | boolean;
  shap_value: number;
  direction: string;
}

export interface UsageContext {
  pitch_usage_overall_pct: number;
  pitch_usage_in_count_pct: number;
  count: string;
  sample_size: number;
}

export interface UpdatedState {
  balls: number;
  strikes: number;
  outs: number;
  on_1b: boolean;
  on_2b: boolean;
  on_3b: boolean;
  at_bat_result: string | null;
}

export interface PredictResponse {
  pitch_outcome: PitchOutcomeResult;
  bip_outcome: BIPOutcomeResult | null;
  explanation: string;
  top_shap_factors: ShapFactor[];
  usage_context: UsageContext;
  updated_state: UpdatedState;
}

export interface PitcherSummary {
  pitcher_id: number;
  full_name: string;
  team: string;
  p_throws: string;
}

export interface ArsenalEntry {
  pitch_type: string;
  pitch_name: string;
  usage_pct: number;
  avg_speed: number;
  avg_spin: number;
  avg_pfx_x: number;
  avg_pfx_z: number;
  sample_size: number;
}

export interface UsageEntry {
  count: string;
  pitch_type: string;
  usage_pct: number;
  sample_size: number;
}

export interface PitchersResponse {
  pitchers: PitcherSummary[];
}

export interface ArsenalResponse {
  pitcher_id: number;
  full_name: string;
  season: number;
  arsenal: ArsenalEntry[];
}

export interface UsageResponse {
  pitcher_id: number;
  usage_by_count: UsageEntry[];
}
