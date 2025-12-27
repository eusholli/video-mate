"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, SystemStatus, Session } from "@/lib/api";
import { formatDistanceToNow } from "date-fns";

export default function Home() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  // Status Badge Helper
  const getStatusBadge = (status: Session["status"]) => {
    switch (status) {
      case "active_process":
      case "indexing":
        return <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-800 animate-pulse">Processing</span>;
      case "ready":
        return <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-800">Ready</span>;
      default:
        return <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-800">Unknown</span>;
    }
  };

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    async function fetchData() {
      try {
        // Fetch System Status
        const statusData = await api.get<SystemStatus>("/api/system/status");
        setStatus(statusData);

        // Fetch Sessions
        const sessionsData = await api.getSessions();
        if (sessionsData.success) {
          setSessions(sessionsData.sessions);
        }

        // 1. Auto-initialize config if needed
        if (!(statusData as any).global_config_set) {
          console.log("System not initialized. Triggering backend initialization...");
          await api.initializeSystem({});
        }

        return statusData;
      } catch (err) {
        console.warn("Backend poll failed", err);
        setStatus(null);
        return null;
      }
    }

    // Initial Trigger & Polling Setup
    const initAndPoll = async () => {
      const initialStatus = await fetchData();
      setLoading(false);

      // If not loaded, trigger load ONCE
      if (initialStatus && initialStatus.global_config_set && !initialStatus.imagebind_loaded) {
        api.loadImageBind().catch(e => console.warn("Background load failed", e));
      }

      // Poll every 5 seconds (slower poll for home page)
      intervalId = setInterval(async () => {
        await fetchData();
      }, 5000);
    };

    initAndPoll();

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="container mx-auto p-8 flex flex-col items-center justify-center gap-12 py-20">
      <header className="text-center space-y-4">
        <h1 className="text-5xl font-light tracking-tight text-primary">Video Mate</h1>
        <p className="text-muted-foreground text-lg">Intelligent Video Analytics & RAG Agent</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">
        <Card className="hover:shadow-md transition-shadow">
          <CardHeader>
            <CardTitle>Ingest Videos</CardTitle>
            <CardDescription>Upload and analyze new video content.</CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/ingest" passHref>
              <Button size="lg" className="w-full">Start Ingestion</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow">
          <CardHeader>
            <CardTitle>Chat Sessions</CardTitle>
            <CardDescription>Interact with your video knowledge base.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">

            <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
              {sessions.length > 0 ? (
                sessions.map((session) => (
                  <Link key={session.chat_id} href={`/chat/${session.chat_id}`} passHref>
                    <div className="flex items-center justify-between p-3 rounded-md border hover:bg-muted/50 transition-colors cursor-pointer mb-2">
                      <div className="flex flex-col">
                        <span className="font-medium text-sm">Session {session.chat_id.substring(0, 8)}...</span>
                        <span className="text-xs text-muted-foreground">
                          {formatDistanceToNow(new Date(session.last_updated * 1000), { addSuffix: true })}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground bg-secondary px-2 py-0.5 rounded-full">
                          {session.video_count} videos
                        </span>
                        {getStatusBadge(session.status)}
                      </div>
                    </div>
                  </Link>
                ))
              ) : (
                <div className="text-sm text-center py-8 text-muted-foreground">
                  No active sessions found. Start by ingesting a video.
                </div>
              )}
            </div>

            <Link href="/ingest" passHref>
              <Button variant="secondary" className="w-full">Create New Session</Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      <div className="flex items-center justify-center gap-6 mt-8 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">System:</span>
          {loading ? (
            <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
          ) : status ? (
            <span className="text-green-600 font-medium">Online</span>
          ) : (
            <span className="text-red-500 font-medium">Offline</span>
          )}
        </div>

        {status && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">AI Model:</span>
            {status.imagebind_loaded ? (
              <span className="text-green-600 font-medium">Ready</span>
            ) : (
              <span className="text-amber-500 font-medium flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                Warming up...
              </span>
            )}
          </div>
        )}

        {status && (
          <div className="text-muted-foreground">
            Total Sessions: {sessions.length}
          </div>
        )}
      </div>
    </div>
  );
}
