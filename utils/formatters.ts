export function formatCurrency(value: number, options: Intl.NumberFormatOptions = {}) {
  if (!Number.isFinite(value)) return "–";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
    ...options,
  }).format(value);
}

export function formatPercent(value: number, digits = 2) {
  if (!Number.isFinite(value)) return "–";
  const percentValue = Math.abs(value) <= 1 ? value * 100 : value;
  return `${percentValue >= 0 ? "+" : ""}${percentValue.toFixed(digits)}%`;
}

export function formatDateLabel(value: string | Date) {
  const date = typeof value === "string" ? new Date(value) : value;
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function downloadTextFile(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function downloadJson(filename: string, data: unknown) {
  downloadTextFile(filename, JSON.stringify(data, null, 2));
}
