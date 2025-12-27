"use server"

import { connectToDatabase } from "@/database/mongoose"
import Watchlist from "@/database/models/watchlist.model"

export const getWatchlistSymbolsByEmail = async (email: string): Promise<string[]> => {
    try {
        const mongoose = await connectToDatabase()
        const db = mongoose.connection.db
        if (!db) throw new Error("Database not connected")

        const user = await db.collection("user").findOne({ email })
        if (!user) return []

        const userId = user.id || (user._id ? user._id.toString() : null)
        if (!userId) return []

        // Use the Watchlist model to query symbols
        const items = await Watchlist.find({ userId }).select("symbol -_id").lean().exec()
        if (!items || items.length === 0) return []

        return items.map(i => (i.symbol || ""))
    } catch (error) {
        console.error("Error fetching watchlist symbols by email", error)
        return []
    }
}

export const getWatchlistItemsByEmail = async (email: string): Promise<{ symbol: string; company: string }[]> => {
    try {
        const mongoose = await connectToDatabase()
        const db = mongoose.connection.db
        if (!db) throw new Error("Database not connected")

        const user = await db.collection("user").findOne({ email })
        if (!user) return []

        const userId = user.id || (user._id ? user._id.toString() : null)
        if (!userId) return []

        const items = await Watchlist.find({ userId }).select("symbol company -_id").lean().exec()
        if (!items || items.length === 0) return []

        return items.map(i => ({ symbol: (i.symbol || "").toUpperCase(), company: i.company || "" }))
    } catch (error) {
        console.error("Error fetching watchlist items by email", error)
        return []
    }
}

export const addSymbolToWatchlist = async (email: string, symbol: string, company: string) => {
    try {
        const mongoose = await connectToDatabase()
        const db = mongoose.connection.db
        if (!db) throw new Error("Database not connected")

        const user = await db.collection("user").findOne({ email })
        if (!user) return []

        const userId = user.id || (user._id ? user._id.toString() : null)
        if (!userId) return []

        const inWatchlist = await Watchlist.findOne({ userId, symbol })
        if (inWatchlist) return { success: false, error: "Symbol already in watchlist" }

        const item = { userId, symbol, company }
        await Watchlist.create(item)
        return { success: true, data: item }
    } catch (error) {
        console.error("Error adding symbol to watchlist", error)
        return { success: false, error: "Error adding symbol to watchlist" }
    }
}

export const removeSymbolFromWatchlist = async (email: string, symbol: string) => {
    try {
        const mongoose = await connectToDatabase()
        const db = mongoose.connection.db
        if (!db) throw new Error("Database not connected")

        const user = await db.collection("user").findOne({ email })
        if (!user) return { success: false, error: "User not found" }

        const userId = user.id || (user._id ? user._id.toString() : null)
        if (!userId) return { success: false, error: "User id not found" }

        const deleted = await Watchlist.findOneAndDelete({ userId, symbol })
        if (!deleted) return { success: false, error: "Symbol not in watchlist" }

        return { success: true }
    } catch (error) {
        console.error("Error removing symbol from watchlist", error)
        return { success: false, error: "Error removing symbol from watchlist" }
    }
}
