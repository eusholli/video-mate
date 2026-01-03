"use client";

import { useState, useEffect } from "react";
import { api, VideoEntry } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { formatDistanceToNow } from "date-fns";
import { Badge } from "@/components/ui/badge";

interface ResearchLibraryProps {
    selectedIds: string[];
    onSelectionChange: (ids: string[]) => void;
}

export function ResearchLibrary({ selectedIds, onSelectionChange }: ResearchLibraryProps) {
    const [videos, setVideos] = useState<VideoEntry[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.getLibrary()
            .then(res => {
                if (res.success) setVideos(res.videos);
            })
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, []);

    const toggleSelection = (id: string) => {
        if (selectedIds.includes(id)) {
            onSelectionChange(selectedIds.filter(v => v !== id));
        } else {
            onSelectionChange([...selectedIds, id]);
        }
    };

    const toggleAll = () => {
        if (selectedIds.length === videos.length) {
            onSelectionChange([]);
        } else {
            onSelectionChange(videos.map(v => v.id));
        }
    };

    return (
        <Card className="h-full flex flex-col bg-background/50 backdrop-blur">
            <CardHeader className="py-4">
                <div className="flex justify-between items-center">
                    <div>
                        <CardTitle className="text-lg">Source Material</CardTitle>
                        <CardDescription>Select videos to analyze.</CardDescription>
                    </div>
                    {videos.length > 0 && (
                        <div className="flex items-center space-x-2">
                            <Checkbox
                                id="select-all"
                                checked={videos.length > 0 && selectedIds.length === videos.length}
                                onCheckedChange={toggleAll}
                            />
                            <label htmlFor="select-all" className="text-xs text-muted-foreground cursor-pointer">All</label>
                        </div>
                    )}
                </div>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto p-2 pt-0 space-y-2">
                {loading && <div className="text-center text-xs p-4">Loading Library...</div>}

                {videos.map(v => (
                    <div
                        key={v.id}
                        className={`flex items-start p-3 border rounded-lg gap-3 transition-colors cursor-pointer ${selectedIds.includes(v.id) ? "bg-primary/10 border-primary/50" : "hover:bg-muted/50"}`}
                        onClick={() => toggleSelection(v.id)}
                    >
                        <Checkbox
                            checked={selectedIds.includes(v.id)}
                            onCheckedChange={() => toggleSelection(v.id)}
                            className="mt-1"
                        />
                        <div className="flex-1 overflow-hidden">
                            <div className="font-medium text-sm truncate" title={v.title}>{v.title}</div>
                            <div className="flex justify-between items-center mt-1">
                                <span className="text-xs text-muted-foreground">{formatDistanceToNow(v.updated_at * 1000)} ago</span>
                                <Badge variant={v.status === "ready" ? "outline" : "secondary"} className="text-[10px] px-1 py-0 h-5">
                                    {v.status}
                                </Badge>
                            </div>
                        </div>
                    </div>
                ))}
            </CardContent>
        </Card>
    );
}
