"use client";

import { cn } from "@/lib/utils";
import { Shield, Scale, Flame } from "lucide-react";

type RiskProfile = "conservative" | "balanced" | "aggressive";

interface RiskProfileOption {
  id: RiskProfile;
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  borderColor: string;
}

const RISK_PROFILES: RiskProfileOption[] = [
  {
    id: "conservative",
    label: "Conservative",
    description: "Lower risk, higher confidence thresholds",
    icon: <Shield className="size-5" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/50",
  },
  {
    id: "balanced",
    label: "Balanced",
    description: "Moderate risk, balanced thresholds",
    icon: <Scale className="size-5" />,
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/10",
    borderColor: "border-yellow-500/50",
  },
  {
    id: "aggressive",
    label: "Aggressive",
    description: "Higher risk, lower confidence thresholds",
    icon: <Flame className="size-5" />,
    color: "text-orange-400",
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500/50",
  },
];

interface RiskProfileSelectorProps {
  selected: RiskProfile;
  onChange: (profile: RiskProfile) => void;
  disabled?: boolean;
  variant?: "cards" | "buttons";
}

export function RiskProfileSelector({
  selected,
  onChange,
  disabled,
  variant = "cards",
}: RiskProfileSelectorProps) {
  if (variant === "buttons") {
    return (
      <div className="flex items-center gap-1 bg-gray-800/50 rounded-lg p-1">
        {RISK_PROFILES.map((profile) => (
          <button
            key={profile.id}
            type="button"
            disabled={disabled}
            onClick={() => onChange(profile.id)}
            className={cn(
              "flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all",
              selected === profile.id
                ? cn(profile.bgColor, profile.color)
                : "text-gray-400 hover:text-gray-200",
              disabled && "opacity-50 cursor-not-allowed"
            )}
          >
            {profile.icon}
            <span className="hidden sm:inline">{profile.label}</span>
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-3 gap-3">
      {RISK_PROFILES.map((profile) => (
        <button
          key={profile.id}
          type="button"
          disabled={disabled}
          onClick={() => onChange(profile.id)}
          className={cn(
            "flex flex-col items-center gap-2 p-4 rounded-xl border text-center transition-all",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-500/50",
            selected === profile.id
              ? cn(profile.borderColor, profile.bgColor)
              : "border-gray-700 hover:border-gray-600",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          <div className={cn(
            "size-10 rounded-lg flex items-center justify-center",
            selected === profile.id ? profile.bgColor : "bg-gray-800",
            selected === profile.id ? profile.color : "text-gray-500"
          )}>
            {profile.icon}
          </div>
          <div>
            <p className={cn(
              "font-medium",
              selected === profile.id ? profile.color : "text-gray-300"
            )}>
              {profile.label}
            </p>
            <p className="text-xs text-gray-500 mt-1 hidden sm:block">
              {profile.description}
            </p>
          </div>
        </button>
      ))}
    </div>
  );
}

export type { RiskProfile };
