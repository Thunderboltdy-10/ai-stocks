"use client"
import { redirect, useSearchParams } from 'next/navigation'
import React from 'react'

const Page = () => {
    const searchParams = useSearchParams()
    const symbol = searchParams.get('tvwidgetsymbol')?.split(":")[1]

    if (symbol) redirect(`/stocks/${symbol}`)

    redirect("/")

    return null
}

export default Page