import type {
  ArsenalResponse,
  PitchersResponse,
  PredictRequest,
  PredictResponse,
  UsageResponse,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new ApiError(response.status, detail);
  }
  return response.json();
}

export function getPitchers(): Promise<PitchersResponse> {
  return request("/pitchers");
}

export function getArsenal(pitcherId: number): Promise<ArsenalResponse> {
  return request(`/pitchers/${pitcherId}/arsenal`);
}

export function getUsage(pitcherId: number): Promise<UsageResponse> {
  return request(`/pitchers/${pitcherId}/usage`);
}

export function postPredict(body: PredictRequest): Promise<PredictResponse> {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
