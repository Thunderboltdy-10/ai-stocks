import {Inngest} from "inngest"

export const inngest = new Inngest({
    id: "ai-stocks",
    ai: {gemini: {apiKey: process.env.GEMINI_API_KEY}}
})