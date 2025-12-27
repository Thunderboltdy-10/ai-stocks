"use client"
import { useCallback, useEffect, useRef } from "react"

// Returns a stable debounced function. The provided callback may change
// between renders; the debounced function will always call the latest
// callback, but the debounced function identity stays the same unless
// the delay changes. This prevents effects that depend on the debounced
// function from retriggering every render.
export function useDebounce(callback: () => void, delay: number) {
    const timeoutRef = useRef<NodeJS.Timeout | null>(null)
    const callbackRef = useRef(callback)

    // keep latest callback in a ref so the debounced invoker calls the
    // newest implementation without changing its identity
    useEffect(() => {
        callbackRef.current = callback
    }, [callback])

    return useCallback(() => {
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current)
        }

        timeoutRef.current = setTimeout(() => {
            callbackRef.current()
        }, delay)
    }, [delay])
}