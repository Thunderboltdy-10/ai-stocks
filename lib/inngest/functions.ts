import { getAllUsersForNewsEmail } from "../actions/user.actions";
import { sendNewsSummaryEmail, sendWelcomeEmail } from "../nodemailer";
import { inngest } from "./client";
import { NEWS_SUMMARY_EMAIL_PROMPT, PERSONALIZED_WELCOME_EMAIL_PROMPT } from "./prompts";
import { getWatchlistSymbolsByEmail } from "@/lib/actions/watchlist.actions"
import { getFormattedTodayDate } from "../utils";
import { getNews } from "../actions/finnhub.actions";

export const sendSignUpEmail = inngest.createFunction(
    {id: "sign-up-email"},
    {event: "app/user.created"},
    async ({event, step}) => {
        const userProfile = `
            - Country: ${event.data.country}
            - Investment Goals: ${event.data.investmentGoals}
            - Risk Tolerance: ${event.data.riskTolerance}
            - Preferred Industry: ${event.data.preferredIndustry}
        `

        const prompt = PERSONALIZED_WELCOME_EMAIL_PROMPT.replace("{{userProfile}}", userProfile)

        const response = await step.ai.infer("generate-welcome-intro", {
            model: step.ai.models.gemini({model: "gemini-2.5-flash"}), 
            body: {
                contents: [
                    {
                        role: "user",
                        parts: [
                            {text: prompt}
                        ]
                    }
                ]
            }
        })

        await step.run("send-welcome-email", async() => {
            const part = response.candidates?.[0]?.content?.parts?.[0]
            const introText = (part && "text" in part ? part.text : null) || "Thanks for joining Signalist. You now have the tools to track markets and make smarter moves"

            const {data: {email, name}} = event

            return await sendWelcomeEmail({email, name, intro: introText})
        })

        return {
            success: true,
            message: "Welcome email sent successfully"
        }
    }
)

export const sendDailyNewsSummary = inngest.createFunction(
    {id: "daily-news-summary"},
    [{event: "app/send.daily.news"}, {cron: "0 12 * * *"}],
    async ({step}) => {
        const users = await step.run("get-all-users", getAllUsersForNewsEmail)

        if (!users || users.length === 0) return {success: false, message: "No users found"}
        
        // Fetch news for all users inside a single step and collect per-user results
        const results = await step.run("fetch-user-news", async () => {
            const perUser: Array<{ user: User; articles: MarketNewsArticle[] }> = []

            for (const user of users) {
                try {
                    const symbols = await getWatchlistSymbolsByEmail(user.email)
                    let articles: MarketNewsArticle[] = []

                    if (symbols && symbols.length > 0) {
                        articles = await getNews(symbols)
                    }

                    if (!articles || articles.length === 0) {
                        // fallback to general market news
                        articles = await getNews()
                    }

                    perUser.push({ user, articles: (articles || []).slice(0, 6) })
                } catch (err) {
                    console.error(`Error fetching news for user ${user.email}`, err)
                    perUser.push({ user, articles: [] })
                }
            }

            return perUser
        })

        // Process the fetched per-user news (placeholder for summarization and sending)
        const userNewsSummaries: {user: User, newsContent: string | null}[] = []

        for (const {user, articles} of results) {
            try {
                const prompt = NEWS_SUMMARY_EMAIL_PROMPT.replace("{{newsData}}", JSON.stringify(articles, null, 2))

                const response = await step.ai.infer(`summarize-news-${user.email}`, {
                    model: step.ai.models.gemini({model: "gemini-2.5-flash"}),
                    body: {
                        contents: [{role: "user", parts: [{text: prompt}]}]
                    }
                })

                const part = response.candidates?.[0]?.content?.parts?.[0]
                const newsContent =  (part &&  "text" in part ? part.text : null) || "No market news"

                userNewsSummaries.push({user, newsContent})
            } catch (error) {
                console.error("Failed to summarise news for", user.email, error)
                userNewsSummaries.push({user, newsContent: null})
            }
        }

        await step.run("send-news-emails", async () => {
            await Promise.all(
                userNewsSummaries.map(async ({user, newsContent}) => {
                    if (!newsContent) return false
                    return await sendNewsSummaryEmail({email: user.email, date: getFormattedTodayDate(), newsContent})
                })
            )
        })

        return { success: true, message: "Daily news summary emails sent successfully" }
    }
)