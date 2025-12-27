"use server"

import { cache } from "react"
import { getWatchlistSymbolsByEmail } from "@/lib/actions/watchlist.actions"
import { getDateRange, validateArticle, formatArticle } from "@/lib/utils"
import { POPULAR_STOCK_SYMBOLS } from "@/lib/constants"

const FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
const NEXT_PUBLIC_FINNHUB_API_KEY = process.env.NEXT_PUBLIC_FINNHUB_API_KEY ||  ""

export const fetchJSON = async (url: string, revalidateSeconds?: number) => {
    const opts: RequestInit & { next?: { revalidate?: number } } = {}

    if (typeof revalidateSeconds === "number") {
        opts.cache = "force-cache"
        opts.next = { revalidate: revalidateSeconds }
    } else {
        opts.cache = "no-store"
    }

    const res = await fetch(url, opts)
    if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(`Fetch failed: ${res.status} ${res.statusText} ${text}`)
    }

    return res.json()
}

export const getNews = async (symbols?: string[]) => {
    try {
        const { from, to } = getDateRange(5)

        if (symbols && symbols.length > 0) {
            const cleaned = Array.from(new Set(symbols.map(s => (s || "").toUpperCase().trim()).filter(Boolean)))
            if (cleaned.length === 0) return []

            const maxArticles = 6
            const collected: MarketNewsArticle[] = []
            const seen = new Set<string>()

            let rounds = 0
            while (collected.length < maxArticles && rounds < 6) {
                for (const symbol of cleaned) {
                    if (collected.length >= maxArticles) break
                    const url = `${FINNHUB_BASE_URL}/company-news?symbol=${encodeURIComponent(symbol)}&from=${from}&to=${to}&token=${NEXT_PUBLIC_FINNHUB_API_KEY}`
                    let data: RawNewsArticle[] = []
                    try {
                        data = await fetchJSON(url, 60 * 30) // cache for 30 minutes
                    } catch (err) {
                        // log and continue to next symbol
                        console.error(`Error fetching company news for ${symbol}`, err)
                        continue
                    }

                    if (!Array.isArray(data) || data.length === 0) continue

                    const valid = data.find(a => validateArticle(a) && !seen.has(a.url || a.headline || String(a.id)))
                    if (valid) {
                        const formatted = formatArticle(valid as RawNewsArticle, true, symbol, collected.length) as MarketNewsArticle
                        const key = formatted.url || formatted.headline || String(formatted.id)
                        if (!seen.has(key)) {
                            seen.add(key)
                            collected.push(formatted)
                        }
                    }
                }
                rounds++
            }

            return collected.sort((a, b) => (b.datetime || 0) - (a.datetime || 0)).slice(0, 6)
        }

        // No symbols provided - fetch general market news
        const generalUrl = `${FINNHUB_BASE_URL}/news?category=general&token=${NEXT_PUBLIC_FINNHUB_API_KEY}`
        const generalData = await fetchJSON(generalUrl, 60 * 30)
        if (!Array.isArray(generalData)) return []

        const seenKeys = new Set<string>()
        const collectedGeneral: MarketNewsArticle[] = []

        for (let i = 0; i < generalData.length && collectedGeneral.length < 6; i++) {
            const article = generalData[i]
            if (!validateArticle(article)) continue
            const key = (article.id ? String(article.id) : article.url) || article.headline
            if (!key || seenKeys.has(key)) continue
            seenKeys.add(key)
            collectedGeneral.push(formatArticle(article, false, undefined, collectedGeneral.length))
        }

        return collectedGeneral.slice(0, 6)
    } catch (error) {
        console.error("Error in getNews", error)
        throw new Error("Failed to fetch news")
    }
}

type FinnHubSearchResult = {
    symbol: string
    description?: string
    displaySymbol?: string
    type?: string
    exchange?: string
}

type FinnHubSearchResponse = {
    count?: number
    result?: FinnHubSearchResult[]
}

type StockWithWatchlistStatus = {
    symbol: string
    name: string
    exchange: string
    type: string
    isInWatchlist: boolean
}

// cached base search that only depends on the external API response
const searchStocksBase = cache(async (query?: string): Promise<StockWithWatchlistStatus[]> => {
    try {
        let results: FinnHubSearchResult[] = []

        if (!query || query.trim().length === 0) {
            const symbols = POPULAR_STOCK_SYMBOLS.slice(0, 10)
            const profiles = await Promise.all(
                symbols.map((s) => {
                    const url = `${FINNHUB_BASE_URL}/stock/profile2?symbol=${encodeURIComponent(s)}&token=${NEXT_PUBLIC_FINNHUB_API_KEY}`
                    return fetchJSON(url, 60 * 60).catch(() => null)
                })
            )

            results = profiles.map((p, idx) => ({
                symbol: symbols[idx],
                description: p?.name || "",
                displaySymbol: symbols[idx],
                type: "Common Stock",
                exchange: p?.exchange || "US",
            }))
        } else {
            const q = query.trim()
            const url = `${FINNHUB_BASE_URL}/search?q=${encodeURIComponent(q)}&token=${NEXT_PUBLIC_FINNHUB_API_KEY}`
            const res = (await fetchJSON(url, 60 * 30)) as FinnHubSearchResponse | FinnHubSearchResult[]
            if (Array.isArray(res)) {
                results = res as FinnHubSearchResult[]
            } else if ((res as FinnHubSearchResponse).result && Array.isArray((res as FinnHubSearchResponse).result)) {
                results = (res as FinnHubSearchResponse).result as FinnHubSearchResult[]
            } else {
                results = []
            }
        }

        const mapped = results
            .map((r) => ({
                symbol: (r.symbol || "").toUpperCase(),
                name: r.description || "",
                exchange: r.displaySymbol || r.exchange || "US",
                type: r.type || "Stock",
                isInWatchlist: false, // enrichment happens later per-user
            }))
            .slice(0, 15)

        return mapped
    } catch (error) {
        console.error("Error in stock search (base):", error)
        return []
    }
})

// non-cached wrapper that enriches the cached base results with per-user watchlist data
export const searchStocks = async (query?: string, email?: string): Promise<StockWithWatchlistStatus[]> => {
    try {
        const base = await searchStocksBase(query)

        if (!email || email.trim().length === 0) return base

        const watchSet = new Set<string>()
        try {
            const watchSymbols = await getWatchlistSymbolsByEmail(email)
            watchSymbols.forEach((s) => watchSet.add((s || "").toUpperCase()))
        } catch (err) {
            console.error("Error resolving watchlist for email", err)
        }

        return base.map((r) => ({ ...r, isInWatchlist: watchSet.has(r.symbol) }))
    } catch (error) {
        console.error("Error in searchStocks wrapper:", error)
        return []
    }
}


