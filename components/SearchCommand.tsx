"use client"

import React, { useState, useEffect } from "react"
import {
    CommandDialog,
    CommandInput,
    CommandList,
    CommandEmpty,
} from "@/components/ui/command"
import { Button } from "./ui/button"
import { Loader2, Star, TrendingUp } from "lucide-react"
import Link from "next/link"
import { searchStocks } from "@/lib/actions/finnhub.actions"
import { useDebounce } from "@/hooks/useDebounce"
import { addSymbolToWatchlist, removeSymbolFromWatchlist } from "@/lib/actions/watchlist.actions"

export default function SearchCommand({renderAs = "button", label = "Add stock", initialStocks, userEmail} : SearchCommandProps) {
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

    // Keep stocks in sync if parent updates initialStocks
    useEffect(() => {
        setStocks(initialStocks)
    }, [initialStocks])

    const handleSearch = async () => {
        if (!isSearchMode) return setStocks(initialStocks)

        setLoading(true)
        try {
            // pass userEmail so search can be enriched with watchlist membership
            const results = await searchStocks(searchTerm.trim(), userEmail)
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
    }, [searchTerm, debouncedSearch])


    const handleSelectStock = () => {
        setOpen(false)
        setSearchTerm("")
        setStocks(initialStocks)
    }

    const toggleWatchlistForStock = async (stock: StockWithWatchlistStatus) => {
        const wasIn = !!stock.isInWatchlist

        // Optimistic update: flip
        setStocks((prev) => {
            if (!prev) return prev
            return prev.map((s) => (s.symbol === stock.symbol ? { ...s, isInWatchlist: !wasIn } : s))
        })

        if (!userEmail) {
            console.warn("No userEmail available to persist watchlist")
            return
        }

        try {
            if (wasIn) {
                // remove
                const res = await removeSymbolFromWatchlist(userEmail, stock.symbol)
                if (!res || (typeof res === "object" && !(res as { success?: boolean }).success)) {
                    console.error("removeSymbolFromWatchlist failed", res)
                    // revert optimistic
                    setStocks((prev) => {
                        if (!prev) return prev
                        return prev.map((s) => (s.symbol === stock.symbol ? { ...s, isInWatchlist: true } : s))
                    })
                }
            } else {
                // add
                const res = await addSymbolToWatchlist(userEmail, stock.symbol, stock.name)
                if (!res || (typeof res === "object" && !(res as { success?: boolean }).success)) {
                    console.error("addSymbolToWatchlist failed", res)
                    // revert optimistic
                    setStocks((prev) => {
                        if (!prev) return prev
                        return prev.map((s) => (s.symbol === stock.symbol ? { ...s, isInWatchlist: false } : s))
                    })
                }
            }
        } catch (err) {
            console.error("Error toggling watchlist", err)
            // revert optimistic
            setStocks((prev) => {
                if (!prev) return prev
                return prev.map((s) => (s.symbol === stock.symbol ? { ...s, isInWatchlist: wasIn } : s))
            })
        }
    }

    const handleStarClick = (e: React.MouseEvent, stock: StockWithWatchlistStatus) => {
        // Prevent the Link navigation when clicking the star
        e.preventDefault()
        e.stopPropagation()
        void toggleWatchlistForStock(stock)
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
                            <li className="search-count" aria-hidden>
                                {isSearchMode ? "Search results" : "Popular stocks"}
                                {" "}{displayedStocks?.length || 0}
                            </li>
                            {displayedStocks?.map((stock) => (
                                <li key={stock.symbol} className="search-item group">
                                    <Link
                                        href={`/stocks/${stock.symbol}`}
                                        onClick={handleSelectStock}
                                        className="search-item-link"
                                    >
                                        <TrendingUp className="h-4 w-4 text-gray-500" />
                                        <div className="flex-1">
                                            <div className="search-item-name group-hover:text-yellow-500">
                                                {stock.name}
                                            </div>
                                            <div className="text-sm">
                                                {stock.symbol} | {stock.exchange} | {stock.type}
                                            </div>
                                        </div>
                                        <span
                                            role="button"
                                            tabIndex={0}
                                            onClick={(e) => handleStarClick(e, stock)}
                                            onKeyDown={(e) => {
                                                if (e.key === "Enter" || e.key === " ") {
                                                    // emulate click for keyboard users
                                                    e.preventDefault()
                                                    e.stopPropagation()
                                                    void toggleWatchlistForStock(stock)
                                                }
                                            }}
                                            className="ml-2"
                                            title={stock.isInWatchlist ? 'Remove from watchlist' : 'Add to watchlist'}
                                        >
                                            {stock.isInWatchlist ? (
                                                <Star className="h-6 w-5 text-yellow-500" fill="#E8BA40" />
                                            ) : (
                                                <Star className="h-5 w-5 text-gray-400" />
                                            )}
                                        </span>
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
