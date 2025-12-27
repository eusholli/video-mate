"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { api, ChatSession, VideoEntry } from "@/lib/api";
import { formatDistanceToNow } from "date-fns";

export function SessionList() {
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false); // Mode

    // Creation State
    const [newSessionName, setNewSessionName] = useState("");
    const [libraryVideos, setLibraryVideos] = useState<VideoEntry[]>([]);
    const [selectedVideoIds, setSelectedVideoIds] = useState<Set<string>>(new Set());
    const [isSubmitting, setIsSubmitting] = useState(false);

    const fetchSessions = async () => {
        try {
            const res = await api.getSessions();
            if (res.success) setSessions(res.sessions);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const fetchLibrary = async () => {
        try {
            const res = await api.getLibrary();
            if (res.success) setLibraryVideos(res.videos.filter(v => v.status === 'ready'));
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        fetchSessions();
    }, []);

    const handleDelete = async (id: string) => {
        if (!confirm("Delete this session?")) return;
        await api.deleteSession(id);
        fetchSessions();
    };

    const toggleVideoSelection = (id: string) => {
        const next = new Set(selectedVideoIds);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        setSelectedVideoIds(next);
    };

    const handleCreate = async () => {
        if (!newSessionName || selectedVideoIds.size === 0) return;
        setIsSubmitting(true);
        try {
            await api.createSession(newSessionName, Array.from(selectedVideoIds));
            setCreating(false);
            setNewSessionName("");
            setSelectedVideoIds(new Set());
            fetchSessions();
        } catch (err: any) {
            alert(err.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    const openCreateMode = () => {
        setCreating(true);
        fetchLibrary();
    };

    if (creating) {
        return (
            <Card className="h-full flex flex-col">
                <CardHeader>
                    <CardTitle>New Session</CardTitle>
                    <CardDescription>Select videos to include in this chat context.</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
                    <div>
                        <label className="text-sm font-medium">Session Name</label>
                        <input
                            className="w-full border rounded px-3 py-2 mt-1"
                            value={newSessionName}
                            onChange={e => setNewSessionName(e.target.value)}
                            placeholder="e.g. Research Meeting Analysis"
                        />
                    </div>

                    <div className="flex-1 overflow-hidden flex flex-col">
                        <label className="text-sm font-medium mb-2">Select Videos ({selectedVideoIds.size})</label>
                        <div className="flex-1 overflow-y-auto border rounded p-2 space-y-2">
                            {libraryVideos.length === 0 && <div className="text-sm text-muted-foreground p-2">No ready videos found in Library.</div>}
                            {libraryVideos.map(v => (
                                <div
                                    key={v.id}
                                    className={`flex items-center gap-3 p-2 rounded cursor-pointer border ${selectedVideoIds.has(v.id) ? 'bg-primary/10 border-primary' : 'hover:bg-muted'}`}
                                    onClick={() => toggleVideoSelection(v.id)}
                                >
                                    <div className={`w-4 h-4 rounded-full border ${selectedVideoIds.has(v.id) ? 'bg-primary' : ''}`} />
                                    <div className="flex-1 truncate text-sm">{v.title}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="flex justify-end gap-2 pt-2">
                        <Button variant="ghost" onClick={() => setCreating(false)}>Cancel</Button>
                        <Button disabled={isSubmitting || !newSessionName || selectedVideoIds.size === 0} onClick={handleCreate}>
                            {isSubmitting ? "Creating..." : "Create Session"}
                        </Button>
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="h-full flex flex-col">
            <CardHeader className="flex flex-row items-center justify-between">
                <div>
                    <CardTitle>Sessions</CardTitle>
                    <CardDescription>Access previous chats.</CardDescription>
                </div>
                <Button size="sm" onClick={openCreateMode}>+ New Session</Button>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto space-y-2 pr-2">
                {loading && <div className="text-center text-sm p-4">Loading...</div>}
                {!loading && sessions.length === 0 && <div className="text-center text-sm text-muted-foreground p-4">No sessions found.</div>}

                {sessions.map(s => (
                    <div key={s.id} className="flex items-center justify-between p-3 border rounded hover:bg-muted/50 group">
                        <Link href={`/chat/${s.id}`} className="flex-1 min-w-0">
                            <div className="font-medium truncate">{s.name}</div>
                            <div className="text-xs text-muted-foreground">
                                {s.video_ids.length} videos • {formatDistanceToNow(s.last_active * 1000)} ago
                            </div>
                        </Link>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-700"
                            onClick={(e) => { e.stopPropagation(); handleDelete(s.id); }}
                        >
                            Delete
                        </Button>
                    </div>
                ))}
            </CardContent>
        </Card>
    );
}
