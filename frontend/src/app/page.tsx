"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api, SystemStatus } from "@/lib/api";

export default function Home() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    async function checkStatus() {
      try {
        const data = await api.get<SystemStatus>("/api/system/status");
        setStatus(data);

        // 1. Auto-initialize config if needed (now triggers backend to load from .env)
        if (!(data as any).global_config_set) {
          console.log("System not initialized. Triggering backend initialization...");

          await api.initializeSystem({});
          console.log("System initialization command sent.");
          // We don't need to re-fetch immediately, the next poll will catch it
        }
        // 2. If config set but ImageBind not loaded, trigger it
        else if (!data.imagebind_loaded) {
          // Only trigger if we haven't already (simple check to avoid spamming, 
          // though backend handles concurrent calls safely)
          // We can check a sessionStorage flag or just rely on backend idempotency.
          // Backend logs show it returns 200 OK if already loaded.
          // Let's just trigger it once per page load to be safe, or just trust the poll.
          // Actually, if we are polling, we shouldn't trigger load every second.
          // We should trigger it once.
        }

        return data;
      } catch (err) {
        console.warn("Backend poll failed", err);
        setStatus(null); // Reset status to null so UI shows "Offline"
        return null;
      }
    }

    // Initial Trigger & Polling Setup
    const initAndPoll = async () => {
      const initialStatus = await checkStatus();
      setLoading(false); // Set loading to false after the initial check

      // If not loaded, trigger load ONCE
      if (initialStatus && initialStatus.global_config_set && !initialStatus.imagebind_loaded) {
        console.log("Triggering background ImageBind load...");
        api.loadImageBind().catch(e => console.warn("Background load failed", e));
      }

      // Poll every 2 seconds until loaded
      intervalId = setInterval(async () => {
        const currentStatus = await checkStatus();
        if (currentStatus && currentStatus.imagebind_loaded) {
          // clearInterval(intervalId); // Don't stop polling, so we can detect if backend goes offline
          // Maybe slow down polling? For now, keep it simple.
        }
      }, 2000);
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
          <CardContent>
            {/* If we had session list, we could show it here. For now, just a button to active sessions or new one */}
            {status?.sessions && status.sessions.length > 0 ? (
              <div className="space-y-2">
                {status.sessions.map(sessionId => (
                  <Link key={sessionId} href={`/chat/${sessionId}`} passHref>
                    <Button variant="outline" className="w-full justify-start mb-2">Session {sessionId.substring(0, 8)}...</Button>
                  </Link>
                ))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground mb-4">No active sessions found. Start by ingesting a video.</div>
            )}

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
            Sessions: {status.total_sessions}
          </div>
        )}
      </div>
    </div>
  );
}
