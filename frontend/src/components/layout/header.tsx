"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export function Header() {
    const pathname = usePathname();

    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container mx-auto flex h-16 items-center justify-between px-4 sm:px-8">
                <div className="flex items-center gap-8">
                    <Link href="/" className="flex items-center gap-2">
                        <span className="text-xl font-semibold tracking-tight">Video Mate</span>
                    </Link>

                    <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
                        <Link
                            href="/"
                            className={cn("transition-colors hover:text-foreground/80", pathname === "/" ? "text-foreground" : "text-foreground/60")}
                        >
                            Dashboard
                        </Link>

                    </nav>
                </div>

                <div className="flex items-center gap-4">
                    {/* Placeholder for future user menu or extra actions */}
                </div>
            </div>
        </header>
    );
}
