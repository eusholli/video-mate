const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:64451";

type ApiResponse<T> = {
    success: boolean;
    error?: string;
    data?: T;
} & T; // Intersection to handle cases where data is spread in response

export const api = {
    async get<T>(endpoint: string): Promise<T> {
        const res = await fetch(`${BASE_URL}${endpoint}`);
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.error || `API Error: ${res.statusText}`);
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
            throw new Error(errorData.error || `API Error: ${res.statusText}`);
        }
        return res.json();
    },

    async delete<T>(endpoint: string): Promise<T> {
        const res = await fetch(`${BASE_URL}${endpoint}`, {
            method: "DELETE",
        });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.error || `API Error: ${res.statusText}`);
        }
        return res.json();
    },

    async initializeSystem(config: any = {}): Promise<any> {
        return this.post("/api/initialize", config);
    },

    async loadImageBind(): Promise<any> {
        return this.post("/api/imagebind/load", {});
    },

    async getSessions(): Promise<{ success: boolean; sessions: Session[]; count: number }> {
        return this.get("/api/sessions");
    },
};

export interface Session {
    chat_id: string;
    created_at: number;
    last_updated: number;
    status: "indexing" | "ready" | "active_process" | "unknown";
    video_count: number;
}

export interface SystemStatus {
    success: boolean;
    total_sessions: number;
    imagebind_loaded: boolean;
    sessions: string[]; // This is the old list of active session IDs
    global_config_set?: boolean;
}
