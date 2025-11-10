"use client"

import React, { useState, useEffect, useCallback } from "react"
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandItem,
  CommandEmpty,
  CommandGroup,
} from "@/components/ui/command"
import { Button } from "./ui/button"
import { Loader2, Star, TrendingUp } from "lucide-react"
import Link from "next/link"
import { searchStocks } from "@/lib/actions/finnhub.actions"
import { useDebounce } from "@/hooks/useDebounce"

export default function SearchCommand({renderAs = "button", label = "Add stock", initialStocks} : SearchCommandProps) {
    const [open, setOpen] = useState(false)
    const [searchTerm, setSearchTerm] = useState("")
    const [loading, setLoading] = useState(false)
    const [stocks, setStocks] = useState<StockWithWatchlistStatus[]>(initialStocks)

    const isSearchMode = !!searchTerm.trim()
    const displayedStocks = isSearchMode ? stocks : stocks?.slice(0, 10)

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
                e.preventDefault()
                setOpen((v) => !v)
            }
        }
        window.addEventListener("keydown", onKey)
        return () => window.removeEventListener("keydown", onKey)
    }, [])

    const handleSearch = async () => {
        if (!isSearchMode) return setStocks(initialStocks)

        setLoading(true)
        try {
            const results = await searchStocks(searchTerm.trim())
            setStocks(results)
        } catch {
            setStocks([])
        } finally {
            setLoading(false)
        }
    }

    const debouncedSearch = useDebounce(handleSearch, 300)

    useEffect(() => {
        debouncedSearch()
    }, [searchTerm])


    const handleSelectStock = () => {
        setOpen(false)
        setSearchTerm("")
        setStocks(initialStocks)
    }

    return (
        <>
            {renderAs === "text" ? (
                <span onClick={() => setOpen(true)}  className="search-text">
                    {label}
                </span>
            ) : (
                <Button onClick={() => setOpen(true)}  className="search-btn">
                    {label}
                </Button>
            )}
            <CommandDialog open={open} onOpenChange={setOpen}  className="search-dialog text-white">
                <div className="search-field pl-3">
                    <CommandInput
                        placeholder={loading ? "Loading..." : "Search stocks or symbols..."}
                        value={searchTerm}
                        onValueChange={setSearchTerm}
                        className="search-input pl-2 text-white"
                    />
                    {loading && <Loader2 className="search-loader" />}
                </div>
                <CommandList className="search-list">
                    {loading ? (
                        <CommandEmpty className="search-list-empty">Loading stocks...</CommandEmpty>
                    ) : displayedStocks?.length === 0 ? (
                        <div  className="search-list-indicator">
                            {isSearchMode ? "No results found" : "No stocks available"}
                        </div>
                    ) : (
                        <ul>
                            <div className="search-count">
                                {isSearchMode ? "Search results" : "Popular stocks"}
                                {" "}{displayedStocks?.length || 0}
                            </div>
                            {displayedStocks?.map((stock, i) => (
                                <li key={stock.symbol} className="search-item">
                                    <Link
                                        href={`/stocks/${stock.symbol}`}
                                        onClick={handleSelectStock}
                                        className="search-item-link"
                                    >
                                        <TrendingUp className="h-4 w-4 text-gray-500" />
                                        <div className="flex-1">
                                            <div className="search-item-name">
                                                {stock.name}
                                            </div>
                                            <div className="text-sm text-gray-500">
                                                {stock.symbol} | {stock.exchange} | {stock.type}
                                            </div>
                                        </div>
                                        <Star />
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    )}
                </CommandList>
            </CommandDialog>
        </>
    )
}
