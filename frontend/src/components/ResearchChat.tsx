"use client";

import { useState, useRef, useEffect } from "react";
import { api, ChatSession } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, ArrowRight, PlayCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import Link from "next/link";

interface ResearchChatProps {
    selectedIds: string[];
}

interface Message {
    role: "user" | "assistant";
    content: string;
    sources?: any[];
}

export function ResearchChat({ selectedIds }: ResearchChatProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [activeIds, setActiveIds] = useState<string[]>([]); // Ids current session is built on

    const scrollRef = useRef<HTMLDivElement>(null);
    const idsKey = [...selectedIds].sort().join(",");

    // Auto-scroll
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    // Restore Session
    useEffect(() => {
        if (!idsKey) {
            setMessages([]);
            setSessionId(null);
            return;
        }

        const cachedSessionId = sessionStorage.getItem(`research_session_${idsKey}`);
        if (cachedSessionId) {
            if (cachedSessionId === sessionId) return; // Already loaded

            setLoading(true);
            api.getSession(cachedSessionId).then(res => {
                if (res.success) {
                    setSessionId(cachedSessionId);
                    setActiveIds(selectedIds);
                    setMessages(res.history.map((h: any) => ({
                        role: h.role,
                        content: h.content,
                        sources: h.sources
                    })));
                } else {
                    // Invalid
                    sessionStorage.removeItem(`research_session_${idsKey}`);
                    setSessionId(null);
                    setMessages([]);
                }
            }).catch(() => {
                sessionStorage.removeItem(`research_session_${idsKey}`);
                setSessionId(null);
                setMessages([]);
            }).finally(() => setLoading(false));
        } else {
            // Reset if no cached session for this selection
            setSessionId(null);
            setMessages([]);
        }
    }, [idsKey]);

    const handleSend = async () => {
        if (!input.trim()) return;
        if (selectedIds.length === 0) {
            alert("Please select at least one source video.");
            return;
        }

        const userMsg = input;
        setInput("");
        setMessages(prev => [...prev, { role: "user", content: userMsg }]);
        setLoading(true);

        try {
            let currentSessionId = sessionId;

            // Create new session if selection changed or no session
            // naive check: string sort comparison
            const selectionChanged = JSON.stringify([...selectedIds].sort()) !== JSON.stringify([...activeIds].sort());

            if (!currentSessionId || selectionChanged) {
                const sessionRes = await api.createSession("Research Session", selectedIds);
                currentSessionId = sessionRes.session.id;
                setSessionId(currentSessionId);
                setActiveIds(selectedIds);

                // Cache ID
                const key = [...selectedIds].sort().join(",");
                sessionStorage.setItem(`research_session_${key}`, currentSessionId);
            }

            // Send Query
            await api.querySession(currentSessionId!, userMsg);

            // Poll for status
            const poll = setInterval(async () => {
                try {
                    const statusRes = await api.getSessionStatus(currentSessionId!);
                    // Handle both nested and flat structure (robustness)
                    const qs = statusRes.status.query_status || (statusRes.status.status ? statusRes.status : null);

                    if (qs && qs.status === "completed") {
                        clearInterval(poll);
                        setMessages(prev => [...prev, {
                            role: "assistant",
                            content: qs.answer,
                            sources: qs.sources
                        }]);
                        setLoading(false);
                    } else if (qs && qs.status === "error") {
                        clearInterval(poll);
                        setMessages(prev => [...prev, { role: "assistant", content: `Error: ${qs.message}` }]);
                        setLoading(false);
                    }
                } catch (e) {
                    // Don't clear on temporary network error, but if many fail?
                    console.error("Poll error", e);
                    // For now, if we fail to parse, it might be the NaN issue or similar
                    // We should probably stop
                    clearInterval(poll);
                    setMessages(prev => [...prev, { role: "assistant", content: `Error: Failed to retrieve answer (Backend might be sending invalid data)` }]);
                    setLoading(false);
                }
            }, 1000);

        } catch (err: any) {
            console.error(err);
            setMessages(prev => [...prev, { role: "assistant", content: `Failed: ${err.message}` }]);
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-background/50 backdrop-blur rounded-lg border shadow-sm overflow-hidden">
            <ScrollArea className="flex-1 p-4">
                <div className="space-y-6 max-w-4xl mx-auto">
                    {messages.length === 0 && (
                        <div className="text-center text-muted-foreground my-20">
                            <h3 className="text-2xl font-semibold mb-2">Research Assistant</h3>
                            <p>Select videos on the left and ask questions to analyze them.</p>
                        </div>
                    )}

                    {messages.map((m, i) => (
                        <div key={i} className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}>
                            <div className={`max-w-[85%] rounded-lg p-4 ${m.role === "user"
                                ? "bg-primary text-primary-foreground"
                                : "bg-muted/50 border"
                                }`}>
                                <div className="prose dark:prose-invert text-sm max-w-none">
                                    <ReactMarkdown>
                                        {m.content}
                                    </ReactMarkdown>
                                </div>
                            </div>

                            {/* Source Cards */}
                            {m.sources && m.sources.length > 0 && (
                                <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 w-full animate-in fade-in slide-in-from-bottom-2">
                                    {m.sources.map((src: any, j: number) => {
                                        // Attempt to parse ID to get video ID and timestamp
                                        // ID format assumption: "video_id_index" or similar?
                                        // The backend source dict is: {id, type, score}
                                        // We need the BACKEND to enrich this or we parse it here?
                                        // _op.py returns `remain_segments`.
                                        // The ID is key in `video_segments`.
                                        // `videorag_query` returns raw objects. 
                                        // The backend update I made didn't enrich them. 
                                        // But wait, the previous `videorag_query` computed context using `video_name` and `index` derived from `s_id`.

                                        // We need to parse: `video_name_index`. 
                                        // Problem: "video_name" might contain underscores.
                                        // But `_op.py` splits by `_` :-/ 
                                        // Correct logic in `_op.py`: `video_name = '_'.join(s_id.split('_')[:-1])`

                                        // We will do client side parsing or fetch details?
                                        // For MVP, client side parsing using the same logic.

                                        const parts = src.id.split('_');
                                        const index = parts[parts.length - 1];
                                        const videoId = parts.slice(0, -1).join('_');

                                        // Link to Detail Page
                                        // We use query param ?hl=index to highlight relevant part
                                        // Or ?t=... if we knew the time.
                                        // We don't have time here without enriching.
                                        // But the Detail page can load the transcript and find the index!

                                        return (
                                            <Link
                                                key={j}
                                                href={`/researcher/video/${videoId}?segment=${index}`}
                                                className="group block"
                                            >
                                                <div className="border rounded p-3 text-xs bg-background hover:border-primary transition-colors cursor-pointer h-full flex flex-col justify-between">
                                                    <div>
                                                        <div className="font-semibold text-primary truncate mb-1">Source {j + 1}</div>
                                                        <div className="text-muted-foreground line-clamp-2 mb-2">
                                                            {src.type === "visual" ? "[Visual Match]" : "[Text Match]"}
                                                        </div>
                                                    </div>
                                                    <div className="flex items-center text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                                                        <PlayCircle className="w-3 h-3 mr-1" />
                                                        <span>View Context</span>
                                                    </div>
                                                </div>
                                            </Link>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    ))}

                    {loading && (
                        <div className="flex items-center gap-2 text-muted-foreground animate-pulse">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Creating answer...</span>
                        </div>
                    )}
                    <div ref={scrollRef} />
                </div>
            </ScrollArea>

            <div className="p-4 border-t bg-background/50 backdrop-blur">
                <div className="flex gap-2 max-w-4xl mx-auto">
                    <Input
                        placeholder="Ask a question about the selected videos..."
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && handleSend()}
                        disabled={loading}
                        className="bg-background"
                    />
                    <Button onClick={handleSend} disabled={loading}>
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
                    </Button>
                </div>
            </div>
        </div>
    );
}
