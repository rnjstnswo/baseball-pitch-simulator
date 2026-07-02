import { useRef } from "react";

interface Props {
  location: { plate_x: number; plate_z: number } | null;
  onSelect: (plate_x: number, plate_z: number) => void;
  disabled: boolean;
}

// Clickable area = the full valid /predict range (feet, catcher's view).
const X_MIN = -2.0;
const X_MAX = 2.0;
const Z_MIN = 0.5;
const Z_MAX = 5.0;

// Rulebook zone: 17in plate width; vertical bounds are the league medians the
// API feeds the model (api/predict.py LEAGUE_SZ_TOP / LEAGUE_SZ_BOT).
const ZONE_HALF_WIDTH = 17 / 12 / 2;
const SZ_TOP = 3.39;
const SZ_BOT = 1.589;

const SCALE = 90; // px per foot
const W = (X_MAX - X_MIN) * SCALE;
const H = (Z_MAX - Z_MIN) * SCALE;

const toPx = (x: number) => (x - X_MIN) * SCALE;
const toPy = (z: number) => (Z_MAX - z) * SCALE; // SVG y grows downward

export default function StrikeZone({ location, onSelect, disabled }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    if (disabled) return;
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return;
    // Screen → viewBox coordinates via the inverse CTM, exact regardless of
    // how CSS scales the element (the ±0.05 ft accuracy criterion).
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    const plate_x = X_MIN + pt.x / SCALE;
    const plate_z = Z_MAX - pt.y / SCALE;
    onSelect(
      Math.min(X_MAX, Math.max(X_MIN, plate_x)),
      Math.min(Z_MAX, Math.max(Z_MIN, plate_z)),
    );
  }

  const zoneX = toPx(-ZONE_HALF_WIDTH);
  const zoneY = toPy(SZ_TOP);
  const zoneW = ZONE_HALF_WIDTH * 2 * SCALE;
  const zoneH = (SZ_TOP - SZ_BOT) * SCALE;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      onClick={handleClick}
      className={`w-full max-w-sm rounded-md border border-gray-200 bg-[#fcfcfb] ${
        disabled ? "cursor-not-allowed opacity-60" : "cursor-crosshair"
      }`}
      role="img"
      aria-label="Strike zone — click to choose pitch location"
    >
      {/* Zone rectangle with a 3×3 grid (recessive hairlines) */}
      <rect
        x={zoneX}
        y={zoneY}
        width={zoneW}
        height={zoneH}
        fill="none"
        stroke="#c3c2b7"
        strokeWidth={1.5}
      />
      {[1, 2].map((i) => (
        <g key={i} stroke="#e1e0d9" strokeWidth={1}>
          <line
            x1={zoneX + (zoneW * i) / 3}
            y1={zoneY}
            x2={zoneX + (zoneW * i) / 3}
            y2={zoneY + zoneH}
          />
          <line
            x1={zoneX}
            y1={zoneY + (zoneH * i) / 3}
            x2={zoneX + zoneW}
            y2={zoneY + (zoneH * i) / 3}
          />
        </g>
      ))}

      {/* Home plate, for orientation (catcher's view) */}
      <polygon
        points={`
          ${toPx(-ZONE_HALF_WIDTH)},${H - 30}
          ${toPx(ZONE_HALF_WIDTH)},${H - 30}
          ${toPx(ZONE_HALF_WIDTH)},${H - 18}
          ${toPx(0)},${H - 6}
          ${toPx(-ZONE_HALF_WIDTH)},${H - 18}
        `}
        fill="none"
        stroke="#c3c2b7"
        strokeWidth={1}
      />

      {/* Selected location: dot with a surface ring so it reads over the grid */}
      {location && (
        <circle
          cx={toPx(location.plate_x)}
          cy={toPy(location.plate_z)}
          r={6}
          fill="#2a78d6"
          stroke="#fcfcfb"
          strokeWidth={2}
        />
      )}
    </svg>
  );
}
