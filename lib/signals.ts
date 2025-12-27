import { TradeMarker } from "@/types/ai";

export type SignalKind = "buy" | "sell" | "hold";

export interface MarkerClassificationOptions {
  shareThreshold?: number;
  confidenceFloor?: number;
}

export const classifySignal = (
  marker: TradeMarker,
  { shareThreshold = 0, confidenceFloor = 0 }: MarkerClassificationOptions
): SignalKind => {
  const confidence = marker.confidence ?? 0;
  if (confidenceFloor > 0 && confidence < confidenceFloor) {
    return "hold";
  }

  const shareValue = Math.abs(marker.shares ?? 0);
  if (shareThreshold > 0 && shareValue < shareThreshold) {
    return "hold";
  }

  return marker.type;
};

export const applySignalThreshold = (marker: TradeMarker, options: MarkerClassificationOptions): TradeMarker => {
  const classification = classifySignal(marker, options);
  if (classification === marker.type) {
    return marker;
  }
  return {
    ...marker,
    displayType: classification,
  } as TradeMarker & { displayType: SignalKind };
};
