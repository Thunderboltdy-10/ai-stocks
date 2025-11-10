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

export default getWatchlistSymbolsByEmail
