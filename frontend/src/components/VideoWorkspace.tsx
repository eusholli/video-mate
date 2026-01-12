"use client";

import { useState, useEffect, useRef } from "react";
import { api, ChatSession, VideoEntry } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, PlayCircle, MessageSquare, Layout, Video } from "lucide-react";
import { TranscriptViewer } from "./TranscriptViewer";
import { ResearchChat } from "./ResearchChat";
import { cn } from "@/lib/utils";

interface VideoWorkspaceProps {
    sessionId: string;
}

export function VideoWorkspace({ sessionId }: VideoWorkspaceProps) {
    const [session, setSession] = useState<ChatSession | null>(null);
    const [videos, setVideos] = useState<VideoEntry[]>([]);
    const [loading, setLoading] = useState(true);

    // State
    const [activeVideoId, setActiveVideoId] = useState<string | null>(null);
    const [currentTime, setCurrentTime] = useState(0);
    const videoRef = useRef<HTMLVideoElement>(null);

    // Tab State: 'transcript' | 'details' (future)
    const [activeTab, setActiveTab] = useState<'transcript'>('transcript');

    useEffect(() => {
        const load = async () => {
            try {
                // 1. Get Session
                const sessionRes = await api.getSession(sessionId);
                if (!sessionRes.success) throw new Error("Failed to load session");
                setSession(sessionRes.session);

                // 2. Get Library (to get video details)
                const libRes = await api.getLibrary();
                if (libRes.success) {
                    const sessionVideos = libRes.videos.filter(v => sessionRes.session.video_ids.includes(v.id));
                    setVideos(sessionVideos);
                    if (sessionVideos.length > 0) {
                        setActiveVideoId(sessionVideos[0].id);
                    }
                }
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };
        load();
    }, [sessionId]);

    const activeVideo = videos.find(v => v.id === activeVideoId);

    const handleSeek = (time: number) => {
        if (videoRef.current) {
            videoRef.current.currentTime = time;
            videoRef.current.play();
        }
    };

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            setCurrentTime(videoRef.current.currentTime);
        }
    };

    const handleClipGenerated = (clip: any) => {
        console.log("Clip generated", clip);
        alert(`Clip generated: ${clip.url}`);
    };

    if (loading) return <div className="flex h-screen items-center justify-center"><Loader2 className="animate-spin text-primary" /></div>;
    if (!session) return <div className="p-8 text-center">Session not found</div>;

    return (
        <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
            {/* Header */}
            <div className="border-b bg-card p-3 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg text-primary">
                        <Layout className="w-5 h-5" />
                    </div>
                    <div>
                        <h1 className="font-semibold text-lg leading-tight">{session.name}</h1>
                        <p className="text-xs text-muted-foreground">{videos.length} videos • Workspace</p>
                    </div>
                </div>
                <div>
                    {/* Toolbar later */}
                </div>
            </div>

            <div className="flex-1 flex min-h-0">
                {/* Sidebar: Videos */}
                <div className="w-64 border-r bg-muted/10 flex flex-col shrink-0">
                    <div className="p-3 border-b text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Project Videos
                    </div>
                    <ScrollArea className="flex-1">
                        <div className="p-2 space-y-1">
                            {videos.map(v => (
                                <button
                                    key={v.id}
                                    onClick={() => setActiveVideoId(v.id)}
                                    className={cn(
                                        "w-full text-left p-2 rounded-lg text-sm transition-colors flex items-start gap-2",
                                        activeVideoId === v.id
                                            ? "bg-primary text-primary-foreground shadow-sm"
                                            : "hover:bg-muted text-muted-foreground hover:text-foreground"
                                    )}
                                >
                                    <Video className="w-4 h-4 mt-0.5 shrink-0" />
                                    <span className="line-clamp-2">{v.title}</span>
                                </button>
                            ))}
                        </div>
                    </ScrollArea>
                    <div className="p-3 border-t">
                        <Button variant="outline" size="sm" className="w-full justify-start gap-2">
                            <span>+ Add Video</span>
                        </Button>
                    </div>
                </div>

                {/* Main Content Areas */}
                <div className="flex-1 flex gap-0 min-w-0">

                    {/* Left: Video & Transcript */}
                    <div className="flex-[1.5] flex flex-col min-w-0 border-r bg-background">
                        {/* Video Player Area */}
                        <div className="aspect-video bg-black relative flex items-center justify-center shrink-0">
                            {activeVideo ? (
                                <video
                                    ref={videoRef}
                                    src={api.getStreamUrl(activeVideo.id)}
                                    controls
                                    className="w-full h-full"
                                    onTimeUpdate={handleTimeUpdate}
                                >
                                    <source src={api.getStreamUrl(activeVideo.id)} type="video/mp4" />
                                </video>
                            ) : (
                                <div className="text-muted-foreground flex flex-col items-center gap-2">
                                    <PlayCircle className="w-8 h-8" />
                                    <span>Select a video</span>
                                </div>
                            )}
                        </div>

                        {/* Transcript Area */}
                        <div className="flex-1 flex flex-col min-h-0 border-t">
                            <div className="flex border-b">
                                <button
                                    className={cn("px-4 py-2 text-sm font-medium border-b-2 transition-colors", activeTab === 'transcript' ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground")}
                                    onClick={() => setActiveTab('transcript')}
                                >
                                    Transcript
                                </button>
                            </div>

                            {activeVideo && activeTab === 'transcript' ? (
                                <TranscriptViewer
                                    videoId={activeVideo.id}
                                    currentTime={currentTime}
                                    onSeek={handleSeek}
                                    onClipGenerated={handleClipGenerated}
                                />
                            ) : (
                                <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
                                    {activeVideo ? "Select a tab" : "Transcript will appear here"}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Right: Chat */}
                    <div className="flex-1 flex flex-col min-w-[350px] bg-muted/5">
                        <div className="p-2 border-b bg-background/50 text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                            <MessageSquare className="w-4 h-4" />
                            AI Assistant
                        </div>
                        <div className="flex-1 min-h-0 relative">
                            <ResearchChat
                                sessionId={sessionId}
                                onSourceClick={(source) => {
                                    // Fallback: RAG video_name might be the ID itself (MD5 hash)
                                    const targetVideoId = source.video_id || source.video_name;
                                    console.log("Source Clicked:", source, "Target ID:", targetVideoId);

                                    if (targetVideoId) {
                                        setActiveVideoId(targetVideoId);
                                        // Wait a tick for video element to update
                                        setTimeout(() => {
                                            if (videoRef.current) {
                                                const timeToSeek = source.start || 0;
                                                console.log("Seeking to:", timeToSeek);
                                                videoRef.current.currentTime = timeToSeek;
                                                videoRef.current.play().catch(e => console.log("Auto-play prevented", e));
                                                setCurrentTime(timeToSeek);
                                            } else {
                                                console.warn("Video ref not found after switch");
                                            }
                                        }, 100);
                                    } else {
                                        console.error("Cannot navigate: No video_id or video_name found in source source");
                                        alert("Could not find the video for this source.");
                                    }
                                }}
                            />
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}
