"use client";

import { useState, useRef, useEffect } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, ArrowRight } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface ResearchChatProps {
    sessionId: string;
    onSourceClick?: (source: any) => void;
}

interface Message {
    role: "user" | "assistant";
    content: string;
    sources?: any[];
}

export function ResearchChat({ sessionId, onSourceClick }: ResearchChatProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    // Poll for updates (e.g. if clips are generated elsewhere or just to keep sync)
    // For now, valid syncing on load and after send

    const scrollRef = useRef<HTMLDivElement>(null);
    const prevLengthRef = useRef(0);
    const lastHistoryStrRef = useRef("");

    // Auto-scroll only when message count increases
    useEffect(() => {
        if (messages.length > prevLengthRef.current) {
            if (scrollRef.current) {
                scrollRef.current.scrollIntoView({ behavior: "smooth" });
            }
        }
        prevLengthRef.current = messages.length;
    }, [messages]);

    const loadHistory = async () => {
        if (!sessionId) return;
        try {
            const res = await api.getSession(sessionId);
            if (res.success) {
                const newMessages = res.history.map((h: any) => ({
                    role: h.role,
                    content: h.content,
                    sources: h.sources
                }));

                const newStr = JSON.stringify(newMessages);
                if (newStr !== lastHistoryStrRef.current) {
                    setMessages(newMessages);
                    lastHistoryStrRef.current = newStr;

                    // Stop loading if the last message is from the assistant
                    if (newMessages.length > 0 && newMessages[newMessages.length - 1].role === "assistant") {
                        setLoading(false);
                    }
                }
            }
        } catch (e) {
            console.error("Failed to load history", e);
        }
    };

    // Initial Load & Poll
    useEffect(() => {
        // Reset refs on session change
        prevLengthRef.current = 0;
        lastHistoryStrRef.current = "";

        loadHistory();
        const interval = setInterval(loadHistory, 3000); // Poll every 3s
        return () => clearInterval(interval);
    }, [sessionId]);


    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = input;
        setInput("");
        // Optimistic update
        setMessages(prev => [...prev, { role: "user", content: userMsg }]);
        setLoading(true);

        try {
            await api.querySession(sessionId, userMsg);

            // Trigger immediate reload check
            setTimeout(loadHistory, 500);

            // Note: We DO NOT set loading(false) here. 
            // We wait for the poller (loadHistory) to find the assistant's response.

        } catch (err: any) {
            console.error(err);
            setMessages(prev => [...prev, { role: "assistant", content: `Failed: ${err.message}` }]);
            setLoading(false); // Only stop loading on error
        }
    };

    return (
        <div className="flex flex-col h-full bg-background/50 backdrop-blur rounded-lg border shadow-sm overflow-hidden">
            <ScrollArea className="flex-1 p-4">
                <div className="space-y-6 max-w-4xl mx-auto">
                    {messages.length === 0 && (
                        <div className="text-center text-muted-foreground my-20">
                            <h3 className="text-xl font-semibold mb-2">Workspace Assistant</h3>
                            <p>Ask questions about the videos in this workspace.</p>
                        </div>
                    )}

                    {messages.map((m, i) => (
                        <div key={i} className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}>
                            <div className={`max-w-[85%] rounded-lg p-4 ${m.role === "user"
                                ? "bg-primary text-primary-foreground"
                                : "bg-muted/50 border"
                                }`}>
                                <div className={`prose text-sm max-w-none ${m.role === "user"
                                    ? "[&_*]:text-primary-foreground"
                                    : "dark:prose-invert"
                                    }`}>
                                    <ReactMarkdown>
                                        {m.content}
                                    </ReactMarkdown>
                                </div>
                            </div>

                            {/* Source Cards */}
                            {m.sources && m.sources.length > 0 && (
                                <div className="mt-4 flex flex-col gap-3 w-full animate-in fade-in slide-in-from-bottom-2">
                                    {m.sources.map((src: any, j: number) => {
                                        // Clean text content
                                        let content = src.content || "";
                                        // Remove metadata prefixes common in RAG output
                                        content = content.replace(/Caption:\s*/g, "").replace(/Transcript:\s*/g, "").replace(/\(Transcript Match\)/g, "").trim();

                                        // Format timestamps
                                        const formatTime = (seconds: number) => {
                                            if (!seconds && seconds !== 0) return "";
                                            const m = Math.floor(seconds / 60);
                                            const s = Math.floor(seconds % 60);
                                            return `${m}:${s.toString().padStart(2, '0')}`;
                                        };
                                        const timeRange = (src.start !== undefined && src.end !== undefined)
                                            ? `${formatTime(src.start)} - ${formatTime(src.end)}`
                                            : "";

                                        return (
                                            <button
                                                key={j}
                                                className="border rounded-lg p-3 text-sm bg-card hover:bg-muted/80 transition-all text-left cursor-pointer active:scale-[0.99] duration-200 shadow-sm group"
                                                onClick={() => {
                                                    console.log("Source clicked:", src);
                                                    if (onSourceClick) onSourceClick(src);
                                                }}
                                            >
                                                <div className="flex items-center justify-between mb-2 pb-2 border-b border-border/50">
                                                    <div className="font-semibold text-primary flex items-center gap-2">
                                                        <span className="bg-primary/10 text-primary px-2 py-0.5 rounded text-xs uppercase tracking-wider">Source {j + 1}</span>
                                                        <span className="text-xs text-muted-foreground font-normal">{src.type === "visual" ? "Visual Match" : "Text Match"}</span>
                                                    </div>
                                                    {timeRange && (
                                                        <div className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">
                                                            {timeRange}
                                                        </div>
                                                    )}
                                                </div>

                                                <div className="text-foreground/90 leading-relaxed whitespace-pre-wrap font-serif text-[0.95rem]">
                                                    "{content}"
                                                </div>

                                                <div className="mt-2 text-xs text-primary/0 group-hover:text-primary/100 transition-colors font-medium flex items-center justify-end gap-1">
                                                    Jump to clip <span className="text-lg leading-none">→</span>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    ))}

                    {loading && (
                        <div className="flex flex-col items-start animate-in fade-in slide-in-from-bottom-2">
                            <div className="bg-muted/50 border rounded-lg p-4 flex items-center gap-1.5 h-10 w-16">
                                <span className="block w-1.5 h-1.5 bg-foreground/60 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                <span className="block w-1.5 h-1.5 bg-foreground/60 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                <span className="block w-1.5 h-1.5 bg-foreground/60 rounded-full animate-bounce"></span>
                            </div>
                        </div>
                    )}
                    <div ref={scrollRef} />
                </div>
            </ScrollArea>

            <div className="p-3 border-t bg-background/50 backdrop-blur">
                <div className="flex gap-2 w-full">
                    <Input
                        placeholder="Ask a question..."
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && handleSend()}
                        disabled={loading}
                        className="bg-background"
                    />
                    <Button onClick={handleSend} disabled={loading} size="icon">
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
                    </Button>
                </div>
            </div>
        </div>
    );
}
