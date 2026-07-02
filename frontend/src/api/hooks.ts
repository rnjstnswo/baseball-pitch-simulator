import { useMutation, useQuery } from "@tanstack/react-query";

import { getArsenal, getPitchers, getUsage, postPredict } from "./client";
import type { PredictRequest } from "./types";

export function usePitchers() {
  return useQuery({ queryKey: ["pitchers"], queryFn: getPitchers });
}

export function useArsenal(pitcherId: number | null) {
  return useQuery({
    queryKey: ["arsenal", pitcherId],
    queryFn: () => getArsenal(pitcherId!),
    enabled: pitcherId !== null,
  });
}

export function useUsage(pitcherId: number | null) {
  return useQuery({
    queryKey: ["usage", pitcherId],
    queryFn: () => getUsage(pitcherId!),
    enabled: pitcherId !== null,
  });
}

export function usePredict() {
  return useMutation({
    mutationFn: (body: PredictRequest) => postPredict(body),
  });
}
