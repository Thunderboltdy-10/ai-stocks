import { auth } from "@/lib/better-auth/auth"
import { headers } from "next/headers"
import { getWatchlistItemsByEmail } from "@/lib/actions/watchlist.actions"
import { POPULAR_STOCK_SYMBOLS } from '@/lib/constants'
import StockWatchlistClient, { Stock as StockType } from "@/components/StockWatchlistClient"
import { Button } from "@/components/ui/button"
import React from "react"

const page = async () => {
    // get current user session
    const session = await auth.api.getSession({ headers: await headers() })
    const email = session?.user?.email || null

    // fetch user's watchlist items (only render those)
    const initialData: StockType[] = []
    const SUGGESTION_THRESHOLD = 5
    let suggestions: string[] = []

    if (email) {
        try {
            const items = await getWatchlistItemsByEmail(email)
            // Map watchlist items into StockType shape. Price/change/etc are placeholders until live data is fetched client-side.
            items.forEach((it, idx) => {
                initialData.push({ id: `${idx}-${it.symbol}`, isWatchlist: true, company: it.company, symbol: it.symbol, price: 0, change: 0, marketCap: '-', peRatio: 0 })
            })

            if (initialData.length < SUGGESTION_THRESHOLD) {
                // build suggestions from popular symbols excluding those already in the watchlist
                const existing = new Set(initialData.map(d => d.symbol))
                suggestions = POPULAR_STOCK_SYMBOLS.filter(s => !existing.has(s)).slice(0, SUGGESTION_THRESHOLD - initialData.length)
            }
        } catch (err) {
            console.error("Error fetching watchlist items in page", err)
        }
    }

    return (
        <div>
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-semibold watchlist-title">Watchlist</h2>
                <div>
                    <Button className="yellow-btn-sm">Add Stock</Button>
                </div>
            </div>

            {/* Render only user's watchlist entries */}
            {initialData.length === 0 ? (
                <div className="watchlist-empty-container">
                    <div className="watchlist-empty">
                        <div className="watchlist-star">☆</div>
                        <div className="empty-title">Your watchlist is empty</div>
                        <div className="empty-description">Add stocks to your watchlist using the search. Below are some suggestions.</div>
                    </div>
                </div>
            ) : (
                <StockWatchlistClient initialData={initialData} userEmail={email} />
            )}

            {/* Suggestions (clearly separated) */}
            {suggestions && suggestions.length > 0 && (
                <div className="mt-6">
                    <h3 className="text-lg font-semibold suggestions-title">Suggestions</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
                        {suggestions.map((s) => (
                            <a key={s} className="news-item" href={`/search?query=${s}`}>
                                <div className="news-tag">Suggestion</div>
                                <div className="news-title">{s}</div>
                                <div className="news-meta">Search for {s} or add it to your watchlist.</div>
                            </a>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

export default page