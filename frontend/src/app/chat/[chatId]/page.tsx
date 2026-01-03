"use client";

import { useParams } from "next/navigation";
import { VideoWorkspace } from "@/components/VideoWorkspace";

export default function ChatPage() {
    const params = useParams();
    const chatId = params.chatId as string;

    return <VideoWorkspace sessionId={chatId} />;
}
