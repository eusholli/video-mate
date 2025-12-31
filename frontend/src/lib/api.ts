const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:64451";

export const api = {
    async get<T>(endpoint: string): Promise<T> {
        const res = await fetch(`${BASE_URL}${endpoint}`);
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.error || `API Error: ${res.statusText} `);
        }
        return res.json();
    },

    async post<T>(endpoint: string, body: any): Promise<T> {
        const res = await fetch(`${BASE_URL}${endpoint}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.error || `API Error: ${res.statusText} `);
        }
        return res.json();
    },

    async delete<T>(endpoint: string): Promise<T> {
        const res = await fetch(`${BASE_URL}${endpoint}`, {
            method: "DELETE",
        });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.error || `API Error: ${res.statusText} `);
        }
        return res.json();
    },

    // --- System ---
    async initializeSystem(): Promise<any> {
        return this.post("/api/initialize", {});
    },

    async checkHealth(): Promise<any> {
        return this.get("/api/health");
    },

    // --- ImageBind ---
    async loadImageBind(): Promise<any> {
        return this.post("/api/imagebind/load", {});
    },
    async getImageBindStatus(): Promise<{ success: boolean, status: { initialized: boolean, loaded: boolean } }> {
        return this.get("/api/imagebind/status");
    },

    // --- Library ---
    async getLibrary(): Promise<{ success: boolean; videos: VideoEntry[] }> {
        return this.get("/api/library");
    },
    async ingestVideo(path: string): Promise<{ success: boolean; status: string; video_id?: string; message?: string }> {
        return this.post("/api/library/ingest", { path });
    },
    async uploadFile(file: File): Promise<{ success: boolean; path: string; filename: string }> {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`${BASE_URL}/api/library/upload`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) {
            const error = await res.json().catch(() => ({}));
            throw new Error(error.error || `Upload failed: ${res.statusText} `);
        }
        return res.json();
    },
    async deleteVideo(id: string): Promise<{ success: boolean }> {
        return this.delete(`/api/library/${id}`);
    },

    // --- Sessions ---
    async getSessions(): Promise<{ success: boolean; sessions: ChatSession[] }> {
        return this.get("/api/sessions");
    },
    async getSession(id: string): Promise<{ success: boolean; session: ChatSession; history: any[]; status: any }> {
        const res = await fetch(`${BASE_URL}/api/sessions/${id}`, { cache: 'no-store' });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.error || `API Error: ${res.statusText}`);
        }
        return res.json();
    },
    async createSession(name: string, video_ids: string[]): Promise<{ success: boolean; session: ChatSession }> {
        return this.post("/api/sessions", { name, video_ids });
    },
    async deleteSession(id: string): Promise<{ success: boolean }> {
        return this.delete(`/api/sessions/${id}`);
    },
    async querySession(id: string, query: string): Promise<{ success: boolean; status: string }> {
        return this.post(`/api/sessions/${id}/query`, { query });
    },
    async getSessionStatus(id: string): Promise<{ success: boolean; status: any }> {
        return this.get(`/api/sessions/${id}/status`);
    },
    async generateClips(id: string, query: string): Promise<{ success: boolean; status: string }> {
        return this.post(`/api/sessions/${id}/clips`, { query });
    },
};

export interface VideoEntry {
    id: string;
    title: string;
    original_path: string;
    status: "processing" | "ready" | "error";
    error?: string;
    progress?: number;
    phase?: string;
    created_at: number;
    updated_at: number;
}

export interface ChatSession {
    id: string;
    name: string;
    video_ids: string[];
    created_at: number;
    last_active: number;
}
