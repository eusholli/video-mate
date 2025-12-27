"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { LibraryList } from "@/components/LibraryList";
import { SessionList } from "@/components/SessionList";

export default function Home() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function checkSystem() {
      try {
        await api.initializeSystem();
        const s = await api.checkHealth(); // Simple check
        // Also load ImageBind
        const ib = await api.getImageBindStatus();
        if (!ib.status.loaded) {
          api.loadImageBind();
        }
        setStatus("online");
      } catch (e) {
        setStatus("offline");
      } finally {
        setLoading(false);
      }
    }
    checkSystem();
  }, []);

  return (
    <div className="container mx-auto p-4 md:p-8 h-screen flex flex-col overflow-hidden">
      <header className="flex justify-between items-center mb-6 shrink-0">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-primary">Video Mate</h1>
          <p className="text-muted-foreground">Intelligent Video Analytics & RAG Agent</p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className={`w-2 h-2 rounded-full ${status === 'online' ? 'bg-green-500' : 'bg-red-500'}`} />
          {loading ? "Connecting..." : (status === 'online' ? "System Ready" : "System Offline")}
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6 min-h-0">
        <div className="flex flex-col min-h-0 bg-background/50 backdrop-blur rounded-lg border shadow-sm">
          <LibraryList />
        </div>
        <div className="flex flex-col min-h-0 bg-background/50 backdrop-blur rounded-lg border shadow-sm">
          <SessionList />
        </div>
      </div>
    </div>
  );
}
