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
    clips?: any[];
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
    const [clipStatus, setClipStatus] = useState<any>(null); // Clip generation status
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const isSendingRef = useRef(false); // Ref for blocking updates

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // Scroll to bottom when new messages arrive
    useEffect(() => {
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
        if (isSendingRef.current) return; // Skip if sending

        try {
            const res = await api.getSession(chatId);
            if (res.success) {
                setMessages(res.history);
                setStatus(res.session);

                const qDetails = res.status;
                const cDetails = (res as any).clip_status; // Type assertion until specific type defined

                const isProcessing = qDetails && qDetails.status === 'processing';
                setProcessing(isProcessing);

                if (isProcessing) {
                    setQueryInfo(qDetails);
                } else {
                    setQueryInfo(null);
                }

                setClipStatus(cDetails);
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
    }, [chatId]);

    const handleSend = async () => {
        if (!input.trim() || processing) return;

        const userMsg = input.trim();
        setInput("");
        setProcessing(true);
        isSendingRef.current = true; // Block updates

        // Optimistic update
        const tempMsg: Message = { role: "user", content: userMsg, timestamp: Date.now() / 1000 };
        setMessages(prev => [...prev, tempMsg]);

        try {
            await api.querySession(chatId, userMsg);
            isSendingRef.current = false; // Unblock
            await loadSession();
        } catch (err: any) {
            console.error(err);
            setProcessing(false);
            isSendingRef.current = false; // Unblock
            alert("Failed to send message");
            loadSession();
        }
    };

    const handleGenerateClips = async () => {
        // Find last user message
        const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
        if (!lastUserMsg) return;

        try {
            // Optimistic set
            setClipStatus({ status: "processing" });
            await api.generateClips(chatId, lastUserMsg.content);
            // Polling will pick up result
        } catch (e) {
            alert("Failed to start clip generation");
            setClipStatus(null);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // Check if we should show clip button
    // Show if: Not processing query, Last message is Assistant, and (ClipStatus is null OR ClipStatus is completed/error)
    const lastMsg = messages.length > 0 ? messages[messages.length - 1] : null;
    const showClipButton = !processing && lastMsg?.role === 'assistant' && !lastMsg?.clips;

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
                    <div key={idx} className="flex flex-col w-full gap-2">
                        <div className={cn("flex w-full", msg.role === "user" ? "justify-end" : "justify-start")}>
                            <div className={cn(
                                "max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm",
                                msg.role === "user"
                                    ? "bg-primary text-primary-foreground font-medium rounded-tr-none"
                                    : "bg-muted text-foreground rounded-tl-none border"
                            )}>
                                <div className={cn("prose prose-sm max-w-none", msg.role === "user" ? "text-primary-foreground [&_p]:text-primary-foreground" : "dark:prose-invert")}>
                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                </div>
                            </div>
                        </div>

                        {/* Render Attached Clips */}
                        {msg.clips && msg.clips.length > 0 && (
                            <div className="w-full max-w-2xl mx-auto space-y-4 pt-2">
                                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider pl-2">Generated Evidence</h3>
                                <div className="grid grid-cols-1 gap-4">
                                    {msg.clips.map((clip: any, i: number) => (
                                        <div key={i} className="border rounded-lg overflow-hidden bg-card shadow-sm">
                                            <div className="aspect-video bg-black relative group">
                                                {/* Use HTML5 Video. Add preload="metadata" */}
                                                <video
                                                    src={clip.url}
                                                    controls
                                                    className="w-full h-full object-contain"
                                                    preload="metadata"
                                                />
                                            </div>
                                            <div className="p-3">
                                                <div className="flex justify-between items-start mb-1">
                                                    <h4 className="font-medium text-sm line-clamp-1" title={clip.title}>{clip.title}</h4>
                                                    <span className="text-xs font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                                                        {Math.round(clip.score * 100)}% Match
                                                    </span>
                                                </div>
                                                <p className="text-xs text-muted-foreground italic line-clamp-2">"{clip.caption}"</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
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

                {/* Active Clip Generation Indicator */}
                <div className="w-full">
                    {clipStatus?.status === 'processing' && (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground p-4 bg-muted/30 rounded-lg animate-pulse">
                            <span className="loading-spinner">🎥</span> Generating precise clips...
                        </div>
                    )}

                    {/* Error State */}
                    {clipStatus?.status === 'error' && (
                        <div className="text-red-500 text-sm p-4 bg-red-50 dark:bg-red-900/10 rounded-lg">
                            Error: {clipStatus.message || "Failed to generate clips"}
                        </div>
                    )}

                    {showClipButton && (!clipStatus || clipStatus.status !== 'processing') && (
                        <div className="flex justify-center mt-6">
                            <Button
                                variant="outline"
                                className="gap-2 border-primary/20 hover:border-primary/50"
                                onClick={handleGenerateClips}
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
                                Generate Relevant Clips
                            </Button>
                        </div>
                    )}
                </div>

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
