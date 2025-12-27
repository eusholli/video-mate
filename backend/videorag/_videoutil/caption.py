import os
import asyncio
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
from io import BytesIO
import base64
from openai import AsyncOpenAI
from .._utils import logger

# --- CONFIGURATION ---
# 1. CRITICAL: Set to 1 for Tier 1 OpenAI accounts (30k TPM limit).
# If you increase this, 2 video segments (11k tokens each) will hit the 30k limit instantly.
MAX_CONCURRENCY = 1

# 2. Increase Timeout (Crucial for Video/Image uploads)
CLIENT_TIMEOUT = 180.0

# 3. Reduce Image Size
IMG_RESIZE_W = 640
IMG_RESIZE_H = 360


def encode_pil_image(pil_image):
    buffer = BytesIO()
    # Reduce quality slightly to 65 to further save bandwidth
    pil_image.save(buffer, format="JPEG", quality=65)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)

    # Resize to smaller dimensions
    frames = [
        Image.fromarray(v.astype("uint8")).resize((IMG_RESIZE_W, IMG_RESIZE_H))
        for v in frames
    ]

    base64_images = []
    for frame in frames:
        base64_image = encode_pil_image(frame)
        base64_images.append(base64_image)
    return base64_images


async def _process_single_caption(
    index, video_frames, segment_transcript, global_config
):
    """
    Process a single video segment caption using a DIRECT OpenAI client
    """
    try:
        api_key = os.environ.get(
            "OPENAI_API_KEY", global_config.get("llm", {}).get("api_key")
        )
        base_url = global_config.get("llm", {}).get(
            "base_url", "https://api.openai.com/v1"
        )
        model = global_config.get("llm", {}).get("model", "gpt-4o")

        # Use 'async with' to ensure the client closes properly
        async with AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=CLIENT_TIMEOUT, max_retries=3
        ) as client:

            content = []
            for frame in video_frames:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                    }
                )

            query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
            content.append({"type": "text", "text": query})

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )

            segment_caption = response.choices[0].message.content
            result = (
                segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                if segment_caption
                else ""
            )
            return index, result

    except Exception as e:
        logger.info(f"❌ Caption failed for segment {index}: {str(e)}")
        return index, ""


async def segment_caption_async(
    video_name,
    video_path,
    segment_index2name,
    transcripts,
    segment_times_info,
    global_config,
):
    """Async caption generation with concurrent processing"""

    logger.info(f"🎬 Extracting frames for {len(segment_index2name)} segments...")
    with VideoFileClip(video_path) as video:
        segment_data = {
            index: {
                "frames": encode_video(video, segment_times_info[index]["frame_times"]),
                "transcript": transcripts[index],
            }
            for index in segment_index2name
        }

    logger.info(
        f"🎨 Starting caption generation for {len(segment_index2name)} segments..."
    )

    # --- CONCURRENCY CONTROL ---
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def limited_process(index):
        async with sem:
            return await _process_single_caption(
                index,
                segment_data[index]["frames"],
                segment_data[index]["transcript"],
                global_config,
            )

    results = await asyncio.gather(
        *[limited_process(index) for index in segment_index2name]
    )

    caption_result = {index: caption for index, caption in results}
    logger.info(
        f"🎉 Caption generation completed! Generated {len(caption_result)} captions successfully."
    )
    return caption_result


def segment_caption(
    video_name,
    video_path,
    segment_index2name,
    transcripts,
    segment_times_info,
    caption_result,
    global_config,
):
    """Worker function for multiprocessing"""
    try:
        result = asyncio.run(
            segment_caption_async(
                video_name,
                video_path,
                segment_index2name,
                transcripts,
                segment_times_info,
                global_config,
            )
        )
        for index, caption in result.items():
            caption_result[index] = caption
    except Exception as e:
        logger.error(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError


def merge_segment_information(
    segment_index2name, segment_times_info, transcripts, captions
):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = "-".join(segment_name.split("-")[-2:])
        inserting_segments[index][
            "content"
        ] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index][
            "frame_times"
        ].tolist()
    return inserting_segments


async def _process_retrieved_segment_caption(
    this_segment,
    refine_knowledge,
    video_path_db,
    video_segments,
    num_sampled_frames,
    global_config,
):
    """
    Process retrieved segment with DIRECT OpenAI client
    """
    video_name = "_".join(this_segment.split("_")[:-1])
    index = this_segment.split("_")[-1]
    segment_transcript = video_segments._data[video_name][index]["transcript"]

    try:
        api_key = os.environ.get(
            "OPENAI_API_KEY", global_config.get("llm", {}).get("api_key")
        )
        base_url = global_config.get("llm", {}).get(
            "base_url", "https://api.openai.com/v1"
        )
        model = global_config.get("llm", {}).get("model", "gpt-4o")

        # Use 'async with' to ensure the client closes properly
        async with AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=CLIENT_TIMEOUT, max_retries=3
        ) as client:

            video_path = video_path_db._data[video_name]
            timestamp = video_segments._data[video_name][index]["time"].split("-")
            start, end = eval(timestamp[0]), eval(timestamp[1])

            with VideoFileClip(video_path) as video:
                frame_times = np.linspace(
                    start, end, num_sampled_frames, endpoint=False
                )
                video_frames = encode_video(video, frame_times)

            query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"

            content = []
            for frame in video_frames:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                    }
                )
            content.append({"type": "text", "text": query})

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            segment_caption = response.choices[0].message.content
            this_caption = (
                segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                if segment_caption
                else ""
            )

            result = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
            return this_segment, result

    except Exception as e:
        logger.info(f"❌ Retrieved caption failed for segment {this_segment}: {str(e)}")
        return (
            this_segment,
            f"Caption:\nError generating caption\nTranscript:\n{segment_transcript}\n\n",
        )


async def retrieved_segment_caption_async(
    refine_knowledge,
    retrieved_segments,
    video_path_db,
    video_segments,
    num_sampled_frames,
    global_config,
):
    """Async retrieved segment caption generation"""

    # --- FIX: HARD LIMIT TO PREVENT TIMEOUTS ---
    # Only process the top 3 most relevant segments.
    # Processing 11 segments sequentially takes >3 mins and crashes the frontend.
    if len(retrieved_segments) > 3:
        logger.info(
            f"⚠️ Throttling: Reducing segments from {len(retrieved_segments)} to 3 to prevent timeout."
        )
        retrieved_segments = list(retrieved_segments)[:3]
    # -------------------------------------------

    logger.info(
        f"🔍 Starting detailed caption for {len(retrieved_segments)} retrieved segments..."
    )

    # CRITICAL: Keep this at 1 for Tier 1 accounts
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def limited_process(this_segment):
        async with sem:
            return await _process_retrieved_segment_caption(
                this_segment,
                refine_knowledge,
                video_path_db,
                video_segments,
                num_sampled_frames,
                global_config,
            )

    results = await asyncio.gather(
        *[limited_process(this_segment) for this_segment in retrieved_segments]
    )

    caption_result = {segment_id: caption for segment_id, caption in results}
    logger.info(
        f"🎉 Retrieved caption generation completed! Generated {len(caption_result)} captions successfully."
    )
    return caption_result
