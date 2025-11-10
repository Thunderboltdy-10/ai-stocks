"use client"
import React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"

interface Props {
  symbol: string
  company?: string
}

export default function WatchlistButton({ symbol, company }: Props) {
  const [saved, setSaved] = useState(false)
  const [loading, setLoading] = useState(false)

  const toggle = async () => {
    setLoading(true)
    try {
      // Placeholder: try calling an API if implemented. If not present this will fail silently.
      await fetch(`/api/watchlist`, {
        method: saved ? "DELETE" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, company }),
      })
      setSaved(!saved)
    } catch (err) {
      // noop - keep it local if backend not implemented
      console.warn("Watchlist API unavailable", err)
      setSaved(!saved)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button
      size="sm"
      variant={saved ? "destructive" : "default"}
      onClick={toggle}
      disabled={loading}
      className="flex items-center gap-2"
      aria-pressed={saved}
      title={saved ? "Remove from watchlist" : "Add to watchlist"}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill={saved ? "currentColor" : "none"} xmlns="http://www.w3.org/2000/svg">
        <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" stroke="currentColor" strokeWidth="0" />
      </svg>
      {saved ? "Saved" : "Save"}
    </Button>
  )
}
