"use client";

import { useState } from "react";
import { ResearchLibrary } from "@/components/ResearchLibrary";
import { ResearchChat } from "@/components/ResearchChat";

export default function ResearcherPage() {
    const [selectedIds, setSelectedIds] = useState<string[]>([]);

    return (
        <div className="container mx-auto p-4 md:p-8 h-[calc(100vh-4rem)] flex flex-col overflow-hidden">
            <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-6 min-h-0">
                <div className="md:col-span-4 lg:col-span-3 flex flex-col min-h-0">
                    <ResearchLibrary
                        selectedIds={selectedIds}
                        onSelectionChange={setSelectedIds}
                    />
                </div>
                <div className="md:col-span-8 lg:col-span-9 flex flex-col min-h-0">
                    <ResearchChat selectedIds={selectedIds} />
                </div>
            </div>
        </div>
    );
}
