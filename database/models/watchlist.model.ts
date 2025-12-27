import mongoose, { Document, model, models, Schema } from "mongoose"

export interface WatchlistItem extends Document {
    userId: string
    symbol: string
    company: string
    addedAt: Date
}

const WatchlistSchema = new Schema<WatchlistItem>({
    userId: { type: String, required: true, index: true },
    symbol: { type: String, required: true, uppercase: true, trim: true },
    company: { type: String, required: true, trim: true },
    addedAt: { type: Date, default: Date.now }
}, {
    timestamps: false
})

// Compound unique index to prevent the same user adding the same symbol twice
WatchlistSchema.index({ userId: 1, symbol: 1 }, { unique: true })

export const Watchlist = (models?.Watchlist as mongoose.Model<WatchlistItem>) || model<WatchlistItem>("Watchlist", WatchlistSchema)

export default Watchlist
