export function Footer() {
    return (
        <footer className="border-t bg-muted/40">
            <div className="container mx-auto flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0 px-4 sm:px-8">
                <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
                    <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
                        Built for <strong>Video Mate</strong>.
                    </p>
                </div>
                <div className="text-sm text-muted-foreground">
                    &copy; {new Date().getFullYear()} Video Mate
                </div>
            </div>
        </footer>
    );
}
