"use client";

import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader2, Scissors, X, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import { createPortal } from "react-dom";

interface TranscriptViewerProps {
    videoId: string;
    currentTime: number;
    onSeek: (time: number) => void;
    onClipGenerated: (clip: any) => void;
}

export function TranscriptViewer({ videoId, currentTime, onSeek, onClipGenerated }: TranscriptViewerProps) {
    const [transcript, setTranscript] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [generating, setGenerating] = useState(false);

    // Selection State
    const [selectionRange, setSelectionRange] = useState<{ start: number, end: number, text: string } | null>(null);
    const [generatedClip, setGeneratedClip] = useState<any | null>(null);

    useEffect(() => {
        setLoading(true);
        api.getTranscript(videoId)
            .then(res => {
                if (res.success) setTranscript(res.transcript);
                else setError("Transcript not available");
            })
            .catch(err => {
                console.error(err);
                setError("Failed to load transcript");
            })
            .finally(() => setLoading(false));
    }, [videoId]);

    // Active Sentence Logic
    const activeIndex = transcript.findIndex((s, i) => {
        const next = transcript[i + 1];
        if (next) return currentTime * 1000 >= s.start && currentTime * 1000 < next.start;
        return currentTime * 1000 >= s.start;
    });

    const scrollRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (activeIndex !== -1 && scrollRef.current && !selectionRange) {
            // Only scroll if we are not actively selecting
            const el = scrollRef.current.children[activeIndex] as HTMLElement;
            if (el) {
                el.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        }
    }, [activeIndex, selectionRange]);

    // Handle Text Selection
    useEffect(() => {
        const handleSelectionChange = () => {
            const selection = window.getSelection();
            if (!selection || selection.isCollapsed) {
                // Don't clear immediately on click-to-seek, wait for explicit clear?
                // Actually, standard behavior is clicking elsewhere clears selection.
                // setSelectionRange(null);
                return;
            }

            // We iterate over the nodes to find data-time attributes
            // This is a bit tricky with mixed content.
            // Simplified approach: Look at anchor and focus nodes.

            const range = selection.getRangeAt(0);
            const container = scrollRef.current;
            if (!container || !container.contains(range.commonAncestorContainer)) return;

            // Helper to find closest data attribute
            const getTimeFromNode = (node: Node | null): number | null => {
                if (!node) return null;
                if (node.nodeType === Node.TEXT_NODE) node = node.parentElement;
                if (node instanceof HTMLElement) {
                    const start = node.closest('[data-start]')?.getAttribute('data-start');
                    return start ? parseFloat(start) : null;
                }
                return null;
            };

            // Helper to find closest end data attribute
            const getEndTimeFromNode = (node: Node | null): number | null => {
                if (!node) return null;
                if (node.nodeType === Node.TEXT_NODE) node = node.parentElement;
                if (node instanceof HTMLElement) {
                    const end = node.closest('[data-end]')?.getAttribute('data-end');
                    return end ? parseFloat(end) : null;
                }
                return null;
            };

            // Identify start/end strictly among selected nodes
            // We can assume the order in DOM matches time
            let start = getTimeFromNode(range.startContainer);
            let end = getEndTimeFromNode(range.endContainer);

            // Refinement: If start is null, maybe walk next sibling? 
            // If end is null, walk prev sibling?

            if (start !== null && end !== null) {
                if (start > end) [start, end] = [end, start]; // Swap if backward selection
                setSelectionRange({
                    start: start / 1000,
                    end: end / 1000,
                    text: selection.toString()
                });
            }
        };

        const onMouseUp = () => {
            // Delay slightly to let selection settle
            setTimeout(handleSelectionChange, 10);
        };

        // We listen on the container
        const container = scrollRef.current;
        if (container) {
            container.addEventListener('mouseup', onMouseUp);
        }

        return () => {
            if (container) container.removeEventListener('mouseup', onMouseUp);
        }
    }, [transcript]);

    const clearSelection = () => {
        const sel = window.getSelection();
        if (sel) sel.removeAllRanges();
        setSelectionRange(null);
    };

    const handleGenerateClip = async () => {
        if (!selectionRange) return;
        setGenerating(true);
        try {
            const res = await api.generateManualClip(videoId, selectionRange.start, selectionRange.end);
            if (res.success) {
                // onClipGenerated(res.clip); // Optional: notify parent
                setGeneratedClip(res.clip);
                clearSelection();
            }
        } catch (e) {
            alert("Failed to generate clip");
        } finally {
            setGenerating(false);
        }
    };

    if (loading) return <div className="flex justify-center p-8"><Loader2 className="animate-spin" /></div>;
    if (error) return <div className="text-center p-8 text-muted-foreground">{error}</div>;

    return (
        <div className="flex flex-col h-full bg-background/50 backdrop-blur relative">
            {/* Header / Info */}
            <div className="p-2 border-b flex justify-between items-center bg-card/50 shrink-0 min-h-[40px]">
                <span className="text-xs font-medium text-muted-foreground">Please highlight text to create a clip</span>

                {/* Floating Menu Replacement (Docked Top) */}
                {selectionRange && (
                    <div className="flex items-center gap-2 animate-in fade-in slide-in-from-top-1 bg-primary text-primary-foreground px-3 py-1 rounded-full shadow-lg z-10">
                        <span className="text-xs font-medium">{(selectionRange.end - selectionRange.start).toFixed(1)}s</span>
                        <div className="h-3 w-px bg-primary-foreground/20" />
                        <button
                            className="text-xs font-bold hover:underline flex items-center gap-1"
                            onClick={handleGenerateClip}
                            disabled={generating}
                        >
                            {generating && <Loader2 className="w-3 h-3 animate-spin" />}
                            Create Clip
                        </button>
                        <button
                            className="p-1 hover:bg-white/20 rounded-full ml-1"
                            onClick={clearSelection}
                        >
                            <X className="w-3 h-3" />
                        </button>
                    </div>
                )}
            </div>

            <ScrollArea className="flex-1 p-6 relative">
                <div className="text-base leading-relaxed max-w-3xl mx-auto" ref={scrollRef}>
                    {transcript.map((s, idx) => {
                        const hasWords = s.words && s.words.length > 0;

                        // We use data attributes on the WRAPPER vs Words
                        // Ideally we put data-start on the wrapper if we don't have words

                        return (
                            <span
                                key={idx}
                                className="inline group"
                                data-start={s.start} // Fallback
                                data-end={s.end} // Fallback
                            >
                                {hasWords ? (
                                    s.words.map((w: any, wIdx: number) => {
                                        const isWordActive = currentTime * 1000 >= w.start && currentTime * 1000 < w.end;
                                        return (
                                            <span
                                                key={`${idx}-${wIdx}`}
                                                className={cn(
                                                    "transition-colors duration-100 rounded-sm px-0.5 cursor-text",
                                                    isWordActive ? "bg-yellow-200 dark:bg-yellow-800/50 text-foreground font-medium" : ""
                                                )}
                                                data-start={w.start}
                                                data-end={w.end}
                                                // Make individual words clickable for seeking
                                                onDoubleClick={(e) => {
                                                    e.stopPropagation(); // prevent selection clear
                                                    onSeek(w.start / 1000);
                                                }}
                                            >
                                                {w.text}{" "}
                                            </span>
                                        );
                                    })
                                ) : (
                                    <span
                                        className={cn(
                                            "cursor-text hover:bg-muted/50 rounded px-1",
                                            currentTime * 1000 >= s.start && currentTime * 1000 < s.end ? "text-primary font-medium" : ""
                                        )}
                                        data-start={s.start}
                                        data-end={s.end}
                                        onDoubleClick={() => onSeek(s.start / 1000)}
                                    >
                                        {s.text}{" "}
                                    </span>
                                )}
                            </span>
                        );
                    })}
                </div>
            </ScrollArea>

            {/* Video Modal */}
            {generatedClip && (
                <div className="absolute inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm animate-in fade-in">
                    <div className="bg-card border shadow-xl rounded-xl p-4 max-w-md w-full mx-4 space-y-4">
                        <div className="flex justify-between items-center">
                            <h3 className="font-semibold text-sm">Clip Generated</h3>
                            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setGeneratedClip(null)}>
                                <X className="w-4 h-4" />
                            </Button>
                        </div>

                        <div className="aspect-video bg-black rounded overflow-hidden relative">
                            <video
                                src={generatedClip.url}
                                controls
                                autoPlay
                                className="w-full h-full"
                            />
                        </div>

                        <div className="flex justify-end gap-2">
                            <Button size="sm" variant="outline" onClick={() => setGeneratedClip(null)}>
                                Close
                            </Button>
                            <a href={generatedClip.url} download={`clip_${videoId}.mp4`}>
                                <Button size="sm" className="gap-2">
                                    <Download className="w-4 h-4" />
                                    Download
                                </Button>
                            </a>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
