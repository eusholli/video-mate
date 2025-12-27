"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { uploadFile } from "./actions";
import { api } from "@/lib/api";

export default function IngestPage() {
    const router = useRouter();
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
        }
    };

    const handleIngest = async () => {
        if (!file) {
            setError("Please select a video file.");
            return;
        }
        setUploading(true);
        setError(null);

        try {
            // 1. Upload file to Next.js server to get local path
            const formData = new FormData();
            formData.append("file", file);
            const { path } = await uploadFile(formData);

            // 2. Generate Chat ID
            const chatId = crypto.randomUUID();

            // 3. Call Python Backend to start indexing
            // Note: Backend expects list of paths
            const response = await api.post<any>(`/api/sessions/${chatId}/videos/upload`, {
                video_path_list: [path],
            });

            if (response.success) {
                // Redirect to Chat page
                router.push(`/chat/${chatId}`);
            } else {
                throw new Error(response.error || "Failed to start processing");
            }

        } catch (err: any) {
            console.error(err);
            setError(err.message || "An unexpected error occurred.");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="container mx-auto flex items-center justify-center p-8 py-20">
            <Card className="w-full max-w-lg">
                <CardHeader>
                    <CardTitle>Upload Video</CardTitle>
                    <CardDescription>Select a video to ingest into your knowledge base.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="grid w-full items-center gap-4">
                        <div className="flex flex-col space-y-1.5">
                            <Input
                                type="file"
                                accept="video/*"
                                onChange={handleFileChange}
                                className="cursor-pointer file:cursor-pointer file:text-primary file:font-semibold hover:file:bg-primary/10"
                            />
                        </div>
                        {error && <p className="text-sm text-destructive">{error}</p>}
                    </div>

                    <Button
                        className="w-full"
                        onClick={handleIngest}
                        disabled={!file || uploading}
                    >
                        {uploading ? "Processing..." : "Start Ingestion"}
                    </Button>

                    <Button variant="ghost" className="w-full" onClick={() => router.push('/')}>
                        Back to Dashboard
                    </Button>
                </CardContent>
            </Card>
        </div>
    );
}
