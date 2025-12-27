import { describe, expect, it } from "vitest";
import { applySignalThreshold, classifySignal } from "@/lib/signals";
import { TradeMarker } from "@/types/ai";

const baseMarker = (overrides: Partial<TradeMarker> = {}): TradeMarker => ({
  date: "2025-11-01",
  price: 100,
  type: "buy",
  shares: 1,
  confidence: 0.5,
  ...overrides,
});

describe("classifySignal", () => {
  it("keeps original type when thresholds are satisfied", () => {
    const result = classifySignal(baseMarker({ type: "sell", shares: 2 }), { shareThreshold: 0.5, confidenceFloor: 0.25 });
    expect(result).toBe("sell");
  });

  it("downgrades to hold when shares are tiny", () => {
    const result = classifySignal(baseMarker({ type: "buy", shares: 0.1 }), { shareThreshold: 0.5 });
    expect(result).toBe("hold");
  });

  it("downgrades to hold when confidence below floor", () => {
    const result = classifySignal(baseMarker({ type: "sell", confidence: 0.1 }), { confidenceFloor: 0.25 });
    expect(result).toBe("hold");
  });

  it("ignores constraints when not provided", () => {
    const result = classifySignal(baseMarker({ type: "sell", shares: 0 }), {});
    expect(result).toBe("sell");
  });
});

describe("applySignalThreshold", () => {
  it("returns original marker when unchanged", () => {
    const marker = baseMarker({ type: "sell", shares: 2 });
    const result = applySignalThreshold(marker, { shareThreshold: 0.5 });
    expect(result).toEqual(marker);
  });

  it("adds displayType when downgraded", () => {
    const marker = baseMarker({ type: "buy", shares: 0.2 });
    const result = applySignalThreshold(marker, { shareThreshold: 0.5 });
    expect(result).toEqual({ ...marker, displayType: "hold" });
  });
});
