"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, TranscriptSegment } from "@/lib/api";
import { TranscriptViewer } from "@/components/TranscriptViewer";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

export default function VideoDetailPage() {
    const params = useParams();
    const router = useRouter();
    const videoId = params.videoId as string;

    const [segments, setSegments] = useState<TranscriptSegment[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!videoId) return;

        const fetchData = async () => {
            try {
                const res = await api.getTranscript(videoId);
                if (res.success) {
                    setSegments(res.transcript);
                }
            } catch (err: any) {
                console.error(err);
                setError("Failed to load transcript. Ensure video has been fully processed.");
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [videoId]);

    return (
        <div className="container mx-auto p-4 md:p-8 h-[calc(100vh-4rem)] flex flex-col">
            <div className="flex items-center gap-4 mb-4">
                <Button variant="ghost" size="icon" onClick={() => router.back()}>
                    <ArrowLeft className="w-4 h-4" />
                </Button>
                <h1 className="text-xl font-semibold">Video Transcript Analysis</h1>
            </div>

            <div className="flex-1 min-h-0 bg-background/50 backdrop-blur rounded-lg border shadow-sm overflow-hidden p-1">
                {loading && (
                    <div className="h-full flex items-center justify-center text-muted-foreground">
                        Loading transcript...
                    </div>
                )}

                {error && (
                    <div className="h-full flex flex-col items-center justify-center text-red-500 gap-2">
                        <p>{error}</p>
                        <Button variant="outline" onClick={() => router.push("/researcher")}>Go Back</Button>
                    </div>
                )}

                {!loading && !error && (
                    <TranscriptViewer videoId={videoId} segments={segments} />
                )}
            </div>
        </div>
    );
}
