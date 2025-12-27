"use server";

import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { cwd } from "process";
import { v4 as uuidv4 } from "uuid";

export async function uploadFile(formData: FormData) {
    const file = formData.get("file") as File;
    if (!file) {
        throw new Error("No file uploaded");
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Create uploads directory if it doesn't exist
    const uploadDir = join(cwd(), "uploads");
    await mkdir(uploadDir, { recursive: true });

    // Create a unique filename to avoid collisions
    const uniqueName = `${uuidv4()}-${file.name}`;
    const path = join(uploadDir, uniqueName);

    await writeFile(path, buffer);

    return { path };
}
