"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { api, VideoEntry } from "@/lib/api";
import { formatDistanceToNow } from "date-fns";

export function LibraryList() {
    const [videos, setVideos] = useState<VideoEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);

    const fetchLibrary = async () => {
        try {
            const res = await api.getLibrary();
            if (res.success) {
                setVideos(res.videos);
            }
        } catch (err) {
            console.error(err);
            setError("Failed to load library");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLibrary();
        // Poll for updates while any video is processing
        const interval = setInterval(() => {
            fetchLibrary();
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);
        try {
            // 1. Upload File
            const uploadRes = await api.uploadFile(file);
            if (!uploadRes.success) throw new Error("Upload failed");

            // 2. Ingest uploaded path
            await api.ingestVideo(uploadRes.path);
            await fetchLibrary();
        } catch (err: any) {
            alert(err.message || "Operation failed");
        } finally {
            setUploading(false);
            // Reset input
            e.target.value = "";
        }
    };

    // Kept for manual resume or debug
    const handleIngestPath = async (path: string) => {
        if (!path) return;
        setUploading(true);
        try {
            await api.ingestVideo(path);
            await fetchLibrary();
        } catch (err: any) {
            alert(err.message);
        } finally {
            setUploading(false);
        }
    };

    const handleDelete = async (id: string) => {
        if (!confirm("Delete this video?")) return;
        await api.deleteVideo(id);
        fetchLibrary();
    };

    return (
        <Card className="h-full flex flex-col">
            <CardHeader>
                <CardTitle>Video Library</CardTitle>
                <CardDescription>Manage your ingested videos.</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden flex flex-col gap-4">
                {/* Ingest Form */}
                <div className="flex gap-2 items-center">
                    <Button
                        variant="outline"
                        className="relative cursor-pointer"
                        disabled={uploading}
                    >
                        {uploading ? "Uploading..." : "Select Video File"}
                        <input
                            type="file"
                            accept="video/*"
                            className="absolute inset-0 opacity-0 cursor-pointer"
                            onChange={handleFileUpload}
                            disabled={uploading}
                        />
                    </Button>
                    <div className="text-xs text-muted-foreground flex-1">
                        {uploading ? "Please wait..." : "Supported: MP4, MOV, MKV"}
                    </div>
                </div>

                {/* List */}
                <div className="flex-1 overflow-y-auto space-y-2 pr-2">
                    {loading && <div className="text-center text-sm p-4">Loading...</div>}
                    {!loading && videos.length === 0 && (
                        <div className="text-center text-sm text-muted-foreground p-4">Library is empty.</div>
                    )}
                    {videos.map(v => (
                        <div key={v.id} className="flex items-center justify-between p-3 border rounded hover:bg-muted/50">
                            <div className="overflow-hidden">
                                <div className="font-medium truncate" title={v.title}>{v.title}</div>
                                <div className="text-xs text-muted-foreground flex gap-2">
                                    <span>{v.status}</span>
                                    <span>• {formatDistanceToNow(v.updated_at * 1000)} ago</span>
                                </div>
                                {v.error && <div className="text-xs text-red-500 truncate" title={v.error}>{v.error}</div>}
                            </div>
                            <div className="flex items-center gap-2">
                                {v.status === "error" || v.status === "processing" ? (
                                    <Button variant="outline" size="sm" onClick={() => handleIngestPath(v.original_path)}>
                                        {v.status === "processing" ? "Check" : "Resume"}
                                    </Button>
                                ) : null}
                                <Button variant="secondary" size="sm" className="text-red-500 hover:text-red-700 hover:bg-red-50" onClick={() => handleDelete(v.id)}>X</Button>
                            </div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
