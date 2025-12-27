import { PredictionResult, ScenarioPoint } from "@/types/ai";

export function calculateDirectionalAccuracy(actual: number[], predicted: number[]): number {
  if (!actual.length || actual.length !== predicted.length) return 0;
  const matches = actual.filter((value, idx) => Math.sign(value) === Math.sign(predicted[idx])).length;
  return matches / actual.length;
}

export function calculateSmape(actual: number[], predicted: number[]): number {
  if (!actual.length || actual.length !== predicted.length) return 0;
  const total = actual.reduce((acc, value, idx) => {
    const denom = (Math.abs(value) + Math.abs(predicted[idx])) / 2 || 1;
    return acc + Math.abs(value - predicted[idx]) / denom;
  }, 0);
  return (100 / actual.length) * total;
}

export function synthesizePricePath(
  startPrice: number,
  predictedReturns: number[],
  positions: number[]
): ScenarioPoint[] {
  const points: ScenarioPoint[] = [];
  let currentPrice = startPrice;

  predictedReturns.forEach((ret, idx) => {
    currentPrice = currentPrice * (1 + ret);
    points.push({
      date: String(idx),
      predictedReturn: ret,
      simulatedPrice: currentPrice,
      position: positions[idx] ?? 0,
    });
  });

  return points;
}

export function averageConfidence(probabilities: Array<{ buy: number; sell: number; hold: number }>) {
  if (!probabilities.length) return { buy: 0, sell: 0, hold: 0 };
  const totals = probabilities.reduce(
    (acc, item) => ({
      buy: acc.buy + item.buy,
      sell: acc.sell + item.sell,
      hold: acc.hold + item.hold,
    }),
    { buy: 0, sell: 0, hold: 0 }
  );
  return {
    buy: totals.buy / probabilities.length,
    sell: totals.sell / probabilities.length,
    hold: totals.hold / probabilities.length,
  };
}

export function toCsv(rows: Array<Record<string, string | number>>): string {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const lines = rows.map((row) => headers.map((key) => row[key]).join(","));
  return [headers.join(","), ...lines].join("\n");
}
