"use client";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Download, FileJson, FileSpreadsheet, Image, FileText } from "lucide-react";

interface ExportMenuProps {
  onExportJSON?: () => void;
  onExportCSV?: () => void;
  onExportPNG?: () => void;
  onExportPDF?: () => void;
  disabled?: boolean;
  variant?: "default" | "outline" | "ghost";
  size?: "default" | "sm" | "lg";
}

export function ExportMenu({
  onExportJSON,
  onExportCSV,
  onExportPNG,
  onExportPDF,
  disabled,
  variant = "outline",
  size = "sm",
}: ExportMenuProps) {
  const hasAnyExport = onExportJSON || onExportCSV || onExportPNG || onExportPDF;

  if (!hasAnyExport) return null;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant={variant} size={size} disabled={disabled} className="border-gray-700">
          <Download className="size-4 mr-2" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-48 bg-gray-900 border-gray-700"
      >
        {onExportJSON && (
          <DropdownMenuItem
            onClick={onExportJSON}
            className="flex items-center gap-2 text-gray-300 focus:text-white focus:bg-gray-800"
          >
            <FileJson className="size-4 text-yellow-400" />
            Export as JSON
          </DropdownMenuItem>
        )}
        {onExportCSV && (
          <DropdownMenuItem
            onClick={onExportCSV}
            className="flex items-center gap-2 text-gray-300 focus:text-white focus:bg-gray-800"
          >
            <FileSpreadsheet className="size-4 text-emerald-400" />
            Export as CSV
          </DropdownMenuItem>
        )}
        {(onExportJSON || onExportCSV) && (onExportPNG || onExportPDF) && (
          <DropdownMenuSeparator className="bg-gray-700" />
        )}
        {onExportPNG && (
          <DropdownMenuItem
            onClick={onExportPNG}
            className="flex items-center gap-2 text-gray-300 focus:text-white focus:bg-gray-800"
          >
            <Image className="size-4 text-blue-400" />
            Export as PNG
          </DropdownMenuItem>
        )}
        {onExportPDF && (
          <DropdownMenuItem
            onClick={onExportPDF}
            className="flex items-center gap-2 text-gray-300 focus:text-white focus:bg-gray-800"
          >
            <FileText className="size-4 text-rose-400" />
            Export as PDF
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
