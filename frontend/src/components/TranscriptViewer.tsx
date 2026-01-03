"use client";

import { useState, useRef, useEffect } from "react";
import { TranscriptSegment, api } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, PlayCircle, Scissors } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useSearchParams } from "next/navigation";

interface TranscriptViewerProps {
    videoId: string;
    segments: TranscriptSegment[];
}

export function TranscriptViewer({ videoId, segments }: TranscriptViewerProps) {
    const [selectedRange, setSelectedRange] = useState<{ start: number, end: number, text: string } | null>(null);
    const [generating, setGenerating] = useState(false);
    const [generatedClipUrl, setGeneratedClipUrl] = useState<string | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const searchParams = useSearchParams();

    // Auto-scroll to highlight if query param exists
    useEffect(() => {
        const hl = searchParams.get("segment");
        if (hl && containerRef.current) {
            const el = containerRef.current.querySelector(`[data-segment-index="${hl}"]`);
            if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    }, [searchParams, segments]);

    const handleSelection = () => {
        const selection = window.getSelection();
        if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
            // Don't clear immediately to allow clicking buttons? 
            // setSelectedRange(null);
            return;
        }

        const range = selection.getRangeAt(0);
        const text = selection.toString().trim();
        if (!text) return;

        // Find start and end nodes
        // We assume each word/sentence is wrapped in a span with data-start/data-end attributes
        // Or we traverse up to find the segment container.

        // Helper to find time from node
        const getTime = (node: Node, isStart: boolean): number | null => {
            let curr: Node | null = node;
            if (curr.nodeType === 3) curr = curr.parentElement;

            while (curr && curr !== containerRef.current) {
                if (curr instanceof HTMLElement) {
                    const t = isStart ? curr.getAttribute("data-start") : curr.getAttribute("data-end");
                    if (t) return parseFloat(t);

                    // Fallback to segment level
                    const tSeg = isStart ? curr.getAttribute("data-segment-start") : curr.getAttribute("data-segment-end");
                    if (tSeg) return parseFloat(tSeg);
                }
                curr = curr.parentNode;
            }
            return null;
        };

        const start = getTime(range.startContainer, true);
        const end = getTime(range.endContainer, false);

        if (start !== null && end !== null) {
            // Ensure sequence
            const startTime = Math.min(start, end);
            const endTime = Math.max(start, end);
            setSelectedRange({ start: startTime, end: endTime, text });
            setGeneratedClipUrl(null); // Reset prev clip
        }
    };

    const generateClip = async () => {
        if (!selectedRange) return;
        setGenerating(true);
        try {
            const res = await api.generateExactClip(videoId, selectedRange.start, selectedRange.end);
            if (res.success) {
                setGeneratedClipUrl(res.url);
            }
        } catch (e) {
            alert("Failed to generate clip");
        } finally {
            setGenerating(false);
        }
    };

    return (
        <Card className="h-full flex flex-col bg-background/50 backdrop-blur" onMouseUp={handleSelection} onKeyUp={handleSelection}>
            <CardHeader className="py-2 pb-0 flex flex-row justify-between items-center bg-card/50">
                <CardTitle className="text-sm font-medium">Transcript</CardTitle>
                {selectedRange && (
                    <div className="flex items-center gap-2 animate-in slide-in-from-top-2">
                        <span className="text-xs text-muted-foreground">
                            {selectedRange.start.toFixed(1)}s - {selectedRange.end.toFixed(1)}s
                        </span>
                        <Button size="sm" onClick={generateClip} disabled={generating}>
                            {generating ? <Loader2 className="w-3 h-3 animate-spin mr-1" /> : <Scissors className="w-3 h-3 mr-1" />}
                            Generate Clip
                        </Button>
                    </div>
                )}
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-sm leading-relaxed" ref={containerRef}>
                <Dialog open={!!generatedClipUrl} onOpenChange={(open: boolean) => !open && setGeneratedClipUrl(null)}>
                    <DialogContent className="sm:max-w-4xl bg-black border-slate-800 p-0 overflow-hidden">
                        <DialogHeader className="p-4 absolute top-0 left-0 w-full z-10 bg-gradient-to-b from-black/80 to-transparent">
                            <DialogTitle className="text-white flex items-center gap-2 text-sm shadow-sm">
                                <PlayCircle className="w-4 h-4 text-primary" /> Generated Clip
                            </DialogTitle>
                        </DialogHeader>
                        <div className="aspect-video w-full bg-black flex items-center justify-center">
                            {generatedClipUrl && (
                                <video src={generatedClipUrl} controls autoPlay className="w-full h-full" />
                            )}
                        </div>
                    </DialogContent>
                </Dialog>

                {segments.map((seg) => {
                    const hl = searchParams.get("segment") === seg.index.toString();

                    // Check for detailed transcript (sentences)
                    // The structure in 'api.ts' is just { ... }
                    // backend: {"sentences": [{"text":..., "start":..., "end":...}]}
                    // api.ts interface: detailed_transcript?: { [key: string]: any }
                    const sentences = seg.detailed_transcript?.sentences as any[];

                    return (
                        <div
                            key={seg.id}
                            data-segment-index={seg.index}
                            data-segment-start={seg.start_time}
                            data-segment-end={seg.end_time}
                            className={`p-2 rounded transition-colors ${hl ? "bg-yellow-500/20 ring-1 ring-yellow-500/50" : "hover:bg-muted/30"}`}
                        >
                            <div className="text-xs text-muted-foreground select-none mb-1">
                                {new Date(seg.start_time * 1000).toISOString().substr(11, 8)}
                            </div>
                            <p>
                                {sentences ? (
                                    sentences.map((s, idx) => (
                                        <span
                                            key={idx}
                                            data-start={s.start / 1000} // ms to s
                                            data-end={s.end / 1000}
                                            className="hover:bg-primary/10 cursor-text"
                                        >
                                            {s.text}{' '}
                                        </span>
                                    ))
                                ) : (
                                    // Fallback to plain text if no sentences
                                    <span
                                        data-start={seg.start_time}
                                        data-end={seg.end_time}
                                    >
                                        {seg.text}
                                    </span>
                                )}
                            </p>
                        </div>
                    );
                })}
            </CardContent>
        </Card>
    );
}
