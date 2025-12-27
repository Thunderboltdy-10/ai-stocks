"use client"

import React, { useState } from "react"
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Star } from "lucide-react"
import { addSymbolToWatchlist, removeSymbolFromWatchlist } from "@/lib/actions/watchlist.actions"

export type Stock = {
  id: string
  isWatchlist: boolean
  company: string
  symbol: string
  price: number
  change: number
  marketCap: string
  peRatio: number
}

type Props = {
  initialData: Stock[]
  userEmail?: string | null
}

export default function StockWatchlistClient({ initialData, userEmail }: Props) {
  const [data, setData] = useState<Stock[]>(initialData)

  const toggleWatch = async (symbol: string) => {
    const row = data.find((d) => d.symbol === symbol)
    if (!row) return
    const wasIn = row.isWatchlist

    // optimistic
    setData((prev) => prev.map((r) => (r.symbol === symbol ? { ...r, isWatchlist: !wasIn } : r)))

    if (!userEmail) {
      console.warn("No user email; cannot persist watchlist")
      return
    }

    try {
      if (wasIn) {
        const res = await removeSymbolFromWatchlist(userEmail, symbol)
        if (!res || (typeof res === "object" && !(res as { success?: boolean }).success)) {
          // revert
          setData((prev) => prev.map((r) => (r.symbol === symbol ? { ...r, isWatchlist: wasIn } : r)))
        } else {
          // notify other clients/components that watchlist changed
          try {
            window.dispatchEvent(new CustomEvent("watchlist:updated", { detail: { symbol, action: "removed" } }))
          } catch {
            // ignore in SSR or environments without window
          }
        }
      } else {
        const res = await addSymbolToWatchlist(userEmail, symbol, row.company)
        if (!res || (typeof res === "object" && !(res as { success?: boolean }).success)) {
          setData((prev) => prev.map((r) => (r.symbol === symbol ? { ...r, isWatchlist: wasIn } : r)))
        } else {
          // notify other clients/components that watchlist changed
          try {
            window.dispatchEvent(new CustomEvent("watchlist:updated", { detail: { symbol, action: "added" } }))
          } catch {
            // ignore in SSR or environments without window
          }
        }
      }
    } catch (err) {
      console.error("Error toggling watchlist", err)
      setData((prev) => prev.map((r) => (r.symbol === symbol ? { ...r, isWatchlist: wasIn } : r)))
    }
  }

  return (
    <div>
      <div className="watchlist-table">
        <Table className="bg-transparent">
          <TableHeader>
            <TableRow>
              <TableHead className="w-8 table-header-row" />
              <TableHead className="table-header-row border-l border-gray-600">Company</TableHead>
              <TableHead className="table-header-row border-l border-gray-600">Symbol</TableHead>
              <TableHead className="table-header-row border-l border-gray-600 text-right">Price</TableHead>
              <TableHead className="table-header-row border-l border-gray-600 text-right">Change</TableHead>
              <TableHead className="table-header-row border-l border-gray-600 text-right">Market Cap</TableHead>
              <TableHead className="table-header-row border-l border-gray-600 text-right">P/E Ratio</TableHead>
              <TableHead className="table-header-row border-l border-gray-600 text-right">Alert</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((row) => (
              <TableRow key={row.id} className="table-row">
                <TableCell>
                  <button onClick={() => toggleWatch(row.symbol)} className="p-1 watchlist-icon-btn" aria-label={row.isWatchlist ? "Remove from watchlist" : "Add to watchlist"}>
                    <Star size={18} fill={row.isWatchlist ? "#E8BA40" : "none"} strokeWidth={1.5} className={row.isWatchlist ? "watchlist-icon-added text-yellow-500" : "star-icon text-gray-400"} />
                  </button>
                </TableCell>
                <TableCell className="table-cell">{row.company}</TableCell>
                <TableCell className="table-cell"><span className="font-bold">{row.symbol}</span></TableCell>
                <TableCell className="table-cell text-right">{new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(row.price)}</TableCell>
                <TableCell className={`table-cell text-right ${row.change >= 0 ? "text-emerald-400" : "text-red-400"}`}>{`${row.change >= 0 ? "+" : ""}${row.change.toFixed(2)}%`}</TableCell>
                <TableCell className="table-cell text-right">{row.marketCap}</TableCell>
                <TableCell className="table-cell text-right">{row.peRatio}</TableCell>
                <TableCell className="table-cell text-right">
                  <Button size="sm" className="orange-btn">Add Alert</Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
