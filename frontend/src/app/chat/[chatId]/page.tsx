"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

type Message = {
    role: "user" | "assistant";
    content: string;
};

type IndexingStatus = {
    success: boolean;
    status: "processing" | "completed" | "error" | "terminated";
    message: string;
    current_step: string;
};

type QueryStatus = IndexingStatus & {
    answer?: string;
    query?: string;
};

export default function ChatPage() {
    const params = useParams();
    const chatId = params.chatId as string;
    const router = useRouter();

    // State
    const [indexingStatus, setIndexingStatus] = useState<IndexingStatus | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [queryProcessing, setQueryProcessing] = useState(false);
    const [queryStatusMessage, setQueryStatusMessage] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Polling for Indexing Status
    useEffect(() => {
        let intervalId: NodeJS.Timeout;

        const checkIndexing = async () => {
            try {
                const status = await api.get<IndexingStatus>(`/api/sessions/${chatId}/status`);
                setIndexingStatus(status);

                if (status.status === "completed" || status.status === "error") {
                    clearInterval(intervalId);
                }
            } catch (err) {
                console.error("Indexing status check failed", err);
            }
        };

        // Initial check
        checkIndexing();
        // Poll every 2 seconds
        intervalId = setInterval(checkIndexing, 2000);

        return () => clearInterval(intervalId);
    }, [chatId]);

    // Polling for Query Status
    useEffect(() => {
        let intervalId: NodeJS.Timeout;

        if (queryProcessing) {
            const checkQuery = async () => {
                try {
                    const status = await api.get<QueryStatus>(`/api/sessions/${chatId}/status?type=query`);

                    if (status.status === "processing") {
                        setQueryStatusMessage(status.message || "Thinking...");
                    } else if (status.status === "completed") {
                        setQueryProcessing(false);
                        setQueryStatusMessage(null);
                        if (status.answer) {
                            setMessages(prev => [...prev, { role: "assistant", content: status.answer! }]);
                        }
                        clearInterval(intervalId);
                    } else if (status.status === "error") {
                        setQueryProcessing(false);
                        setQueryStatusMessage("Error: " + status.message);
                        clearInterval(intervalId);
                    }
                } catch (err) {
                    console.error("Query status check failed", err);
                }
            };

            intervalId = setInterval(checkQuery, 1000);
        }

        return () => clearInterval(intervalId);
    }, [chatId, queryProcessing]);

    // Scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, queryStatusMessage]);

    const handleSend = async () => {
        if (!input.trim() || queryProcessing) return;

        const userMsg = input.trim();
        setMessages(prev => [...prev, { role: "user", content: userMsg }]);
        setInput("");
        setQueryProcessing(true);
        setQueryStatusMessage("Sending query...");

        try {
            await api.post(`/api/sessions/${chatId}/query`, { query: userMsg });
        } catch (err: any) {
            console.error(err);
            setQueryProcessing(false);
            setQueryStatusMessage(null);
            setMessages(prev => [...prev, { role: "assistant", content: "Error: Failed to send query." }]);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // Render Loading / Indexing State
    if (!indexingStatus || indexingStatus.status === "processing") {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background p-8">
                <Card className="w-full max-w-md text-center p-8">
                    <div className="flex flex-col items-center gap-4">
                        <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                        <h2 className="text-xl font-semibold">Analyzing Video</h2>
                        <p className="text-muted-foreground">{indexingStatus?.message || "Initializing..."}</p>
                        <div className="text-xs text-muted-foreground uppercase tracking-widest mt-2">{indexingStatus?.current_step}</div>
                    </div>
                </Card>
            </div>
        );
    }

    // Render Chat Interface
    return (
        <div className="flex flex-col h-screen bg-background text-foreground">
            {/* Header */}
            <header className="border-b bg-card p-4 flex items-center justify-between sticky top-0 z-10">
                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" onClick={() => router.push('/')}>← Back</Button>
                    <h1 className="font-semibold text-lg">Session {chatId.substring(0, 8)}</h1>
                </div>
                <div className="text-sm text-muted-foreground">
                    {indexingStatus.status === "completed" ? "Ready" : "Error"}
                </div>
            </header>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 md:p-8 md:max-w-3xl md:mx-auto w-full">
                {messages.length === 0 && (
                    <div className="text-center text-muted-foreground mt-12">
                        Video analysis complete. Ask me anything about the video!
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div key={idx} className={cn("flex w-full", msg.role === "user" ? "justify-end" : "justify-start")}>
                        <div className={cn(
                            "max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm",
                            msg.role === "user"
                                ? "bg-primary text-primary-foreground rounded-tr-none"
                                : "bg-muted text-foreground rounded-tl-none border"
                        )}>
                            {msg.content}
                        </div>
                    </div>
                ))}

                {queryProcessing && (
                    <div className="flex w-full justify-start">
                        <div className="bg-muted text-muted-foreground rounded-2xl rounded-tl-none px-5 py-3 text-sm flex items-center gap-2">
                            <div className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce delay-75" />
                            <div className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce delay-150" />
                            <div className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce delay-300" />
                            <span className="ml-2 text-xs">{queryStatusMessage}</span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-background border-t sticky bottom-0">
                <div className="max-w-3xl mx-auto flex gap-2">
                    <Input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask a question about the video..."
                        className="flex-1"
                        disabled={queryProcessing}
                        autoFocus
                    />
                    <Button onClick={handleSend} disabled={!input.trim() || queryProcessing}>
                        Send
                    </Button>
                </div>
            </div>
        </div>
    );
}
