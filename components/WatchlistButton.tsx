"use client"
import React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { PlusIcon } from "lucide-react"

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
            className="flex flex-1 items-center gap-2 yellow-btn"
            aria-pressed={saved}
            title={saved ? "Remove from watchlist" : "Add to watchlist"}
        >
            <PlusIcon />
            {saved ? "Remove from Watchlist" : "Add to Watchlist"}
        </Button>
    )
}
