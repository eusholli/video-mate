"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";
import ReactMarkdown from "react-markdown";

type Message = {
    role: "user" | "assistant";
    content: string;
    timestamp?: number;
};

export default function ChatPage() {
    const params = useParams();
    const chatId = params.chatId as string;
    const router = useRouter();

    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [status, setStatus] = useState<any>(null); // Session status/metadata
    const [queryInfo, setQueryInfo] = useState<any>(null); // Current query status
    const [processing, setProcessing] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // Scroll to bottom when new messages arrive
    useEffect(() => {
        // Only scroll if we have messages and the last message is different or length changed
        // Simple heuristic: if length changed.
        // For deeper check we could compare IDs, but length is usually sufficient for chat.
        if (messages.length > 0) {
            scrollToBottom();
        }
    }, [messages.length]);

    // Scroll to bottom when new messages arrive
    useEffect(() => {
        if (messages.length > 0) {
            messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages.length]);

    const loadSession = async () => {
        try {
            const res = await api.getSession(chatId);
            if (res.success) {
                // Only update messages if length changed to avoid jitter, or if we need to sync content
                // But for now, we just set it. The scroll effect protects us from jitter.
                setMessages(res.history);
                setStatus(res.session);

                const qDetails = res.status;
                const isProcessing = qDetails && qDetails.status === 'processing';

                setProcessing(isProcessing);

                if (isProcessing) {
                    setQueryInfo(qDetails);
                } else {
                    setQueryInfo(null);
                }
            }
        } catch (e) {
            console.error("Sync failed", e);
        }
    };

    // Initial Load & Polling
    useEffect(() => {
        loadSession();
        const interval = setInterval(loadSession, 2000);
        return () => clearInterval(interval);
    }, [chatId]); // Removed 'processing' dependency to avoid re-setting interval constantly

    const handleSend = async () => {
        if (!input.trim() || processing) return;

        const userMsg = input.trim();
        setInput("");
        setProcessing(true);

        // Optimistic update
        const tempMsg: Message = { role: "user", content: userMsg, timestamp: Date.now() / 1000 };
        setMessages(prev => [...prev, tempMsg]);

        try {
            // Wait for query to acknowledge
            await api.querySession(chatId, userMsg);
            // Immediately fetch session to confirm status/history
            await loadSession();
        } catch (err: any) {
            console.error(err);
            setProcessing(false);
            alert("Failed to send message");
            // Revert optimistic update? For now just leave it or reload.
            loadSession();
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="flex flex-col h-screen bg-background text-foreground">
            {/* Header */}
            <header className="border-b bg-card p-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
                <div className="flex items-center gap-4">
                    <Button variant="ghost" size="sm" onClick={() => router.push('/')}>← Back</Button>
                    <div>
                        <h1 className="font-semibold text-lg">{status?.name || `Session ${chatId.substring(0, 8)}`}</h1>
                        <div className="text-xs text-muted-foreground flex gap-2">
                            <span>{status?.video_ids?.length || 0} videos</span>
                            {status?.last_active && <span>• Active {formatDistanceToNow(status.last_active * 1000)} ago</span>}
                        </div>
                    </div>
                </div>
                <div className="text-sm">
                    {processing ? (
                        <span className="text-amber-500 animate-pulse flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-amber-500"></span> Thinking...
                        </span>
                    ) : (
                        <span className="text-green-500">Ready</span>
                    )}
                </div>
            </header>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 md:p-8 md:max-w-3xl md:mx-auto w-full">
                {messages.map((msg, idx) => (
                    <div key={idx} className={cn("flex w-full", msg.role === "user" ? "justify-end" : "justify-start")}>
                        <div className={cn(
                            "max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm",
                            msg.role === "user"
                                ? "bg-primary text-primary-foreground rounded-tr-none"
                                : "bg-muted text-foreground rounded-tl-none border"
                        )}>
                            <div className="prose prose-sm dark:prose-invert max-w-none">
                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                            </div>
                        </div>
                    </div>
                ))}

                {processing && (
                    <div className="flex w-full justify-start">
                        <div className="bg-muted text-muted-foreground rounded-2xl rounded-tl-none px-5 py-3 text-sm flex items-center gap-2">
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
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
                        disabled={processing}
                        autoFocus
                    />
                    <Button onClick={handleSend} disabled={!input.trim() || processing}>
                        Send
                    </Button>
                </div>
            </div>
        </div>
    );
}
