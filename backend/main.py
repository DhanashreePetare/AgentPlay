import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog
from quart import Quart, request, jsonify, send_file
from quart_cors import cors
from youtube_transcript_api import YouTubeTranscriptApi
from mistralai import Mistral
from gtts import gTTS
import yt_dlp

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool

from backend.config.settings import (
    GEMINI_API_KEY, MISTRAL_API_KEY, SERPER_API_KEY,
    GEMINI_CHAT_MODEL, GEMINI_LLM_MODEL,
    MISTRAL_MODEL, SIMILARITY_THRESHOLD,
    CHUNK_SIZE_WORDS, TRANSCRIPT_CHUNK_SIZE_CHARS,
    AUDIO_DATA_DIR, AUDIO_SEGMENT_TIMEOUT, AUDIO_TOTAL_TIMEOUT,
    RATE_LIMIT_DELAY, MAX_RETRIES, MAX_CONCURRENT_REQUESTS,
    APP_HOST, APP_PORT, DEBUG
)
from backend.config.cache import (
    cache_transcript, get_cached_transcript,
    cache_summary, get_cached_summary,
    cache_notes, get_cached_notes,
    cache_yt_metadata, get_cached_yt_metadata,
    cache_source_language, get_cached_source_language,
    cache_english_transcript, get_cached_english_transcript,
    ping_redis
)
from backend.config.vector_store import (
    store_embeddings_qdrant, search_qdrant,
    is_video_indexed, ping_qdrant
)
from crew_helper import count_words_and_translate, translate_segment

# ── Disable noisy OpenTelemetry ───────────────────────────────────────────────
try:
    from opentelemetry import trace
    trace.set_tracer_provider(trace.NoOpTracerProvider())
except Exception:
    pass

# ── Structured logging setup ─────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer()
    ]
)
log = structlog.get_logger()

# ── Environment wiring ────────────────────────────────────────────────────────
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Model & tool initialization ───────────────────────────────────────────────
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
llm = LLM(model=GEMINI_LLM_MODEL)
llm_genai = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0.4)
serper_tool = SerperDevTool()

# ── Voice configs ─────────────────────────────────────────────────────────────
VOICE_CONFIGS = {
    'en': 'en', 'hi': 'hi', 'es': 'es', 'fr': 'fr',
    'de': 'de', 'ja': 'ja', 'ko': 'ko', 'zh': 'zh-CN',
    'it': 'it', 'pt': 'pt', 'ru': 'ru', 'nl': 'nl',
    'tr': 'tr', 'pl': 'pl', 'id': 'id', 'th': 'th', 'vi': 'vi'
}

LANGUAGE_MAP = {
    'en': "English", 'hi': "Hindi", 'es': "Spanish",
    'fr': "French", 'de': "German", 'ja': "Japanese",
    'ko': "Korean", 'zh': "Chinese (Mandarin)", 'it': "Italian",
    'pt': "Portuguese", 'ru': "Russian", 'nl': "Dutch",
    'tr': "Turkish", 'pl': "Polish", 'id': "Indonesian",
    'th': "Thai", 'vi': "Vietnamese"
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s+1:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Transcript retrieval
# ─────────────────────────────────────────────────────────────────────────────

async def get_transcript_with_timestamps_async(video_id: str):
    """
    Fetch transcript from YouTube with language fallback.
    Returns (segments, source_language, concatenated_text) or (None, None, None).
    """
    ytt_api = YouTubeTranscriptApi()
    transcript_list = None
    source_language = None

    for lang in ['en', 'hi']:
        try:
            transcript_list = ytt_api.fetch(video_id, languages=[lang])
            source_language = lang
            log.info("transcript.fetched", video_id=video_id, lang=lang)
            break
        except Exception:
            continue

    if transcript_list is None:
        try:
            available = ytt_api.list(video_id)
            for t in available:
                transcript_list = t.fetch()
                source_language = t.language_code
                log.info("transcript.fetched.fallback", video_id=video_id, lang=source_language)
                break
        except Exception as e:
            log.error("transcript.fetch.failed", video_id=video_id, error=str(e))
            return None, None, None

    if transcript_list is None:
        return None, None, None

    raw = transcript_list.to_raw_data()
    segments = []
    full_text = ""

    for i, entry in enumerate(raw):
        start = entry['start']
        duration = entry.get('duration', 0)
        end = start + duration
        if i < len(raw) - 1:
            next_start = raw[i + 1]['start']
            end = min(end, next_start)
        segments.append({
            "Segment": i + 1,
            "Text": entry['text'],
            "Start": format_timestamp(start),
            "End": format_timestamp(end),
            "Duration": format_timestamp(end - start)
        })
        full_text += entry['text'] + " "

    return segments, source_language, full_text.strip()


async def get_or_fetch_transcript(video_id: str):
    """
    Returns transcript dict from Redis cache or fetches from YouTube.
    Raises ValueError if no transcript is available.
    """
    cached = await get_cached_transcript(video_id)
    if cached:
        return cached

    segments, source_lang, full_text = await get_transcript_with_timestamps_async(video_id)
    if segments is None:
        raise ValueError(f"No transcript available for video: {video_id}")

    english_text = full_text
    if source_lang != 'en':
        log.info("transcript.translating", video_id=video_id, from_lang=source_lang)
        english_text = await count_words_and_translate(full_text)

    data = {
        "transcript": segments,
        "source_language": source_lang,
        "original_text": full_text,
        "english_text": english_text
    }

    await cache_transcript(video_id, data)
    await cache_source_language(video_id, source_lang)
    await cache_english_transcript(video_id, english_text)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_transcript(transcript_data: dict, max_words: int = CHUNK_SIZE_WORDS) -> list:
    segments = transcript_data.get("transcript", [])
    if not segments:
        return []

    chunks, current_words, current_start = [], [], None

    for i, seg in enumerate(segments):
        words = seg["Text"].split()
        if current_start is None:
            current_start = seg["Start"]

        if len(current_words) + len(words) > max_words and current_words:
            chunks.append({
                "Text": " ".join(current_words),
                "Start": current_start,
                "End": segments[i - 1]["End"]
            })
            current_words = []
            current_start = seg["Start"]

        current_words.extend(words)

    if current_words:
        chunks.append({
            "Text": " ".join(current_words),
            "Start": current_start,
            "End": segments[-1]["End"]
        })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Summarisation & Notes  (Mistral)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_chunk(chunk: str, mode: str = "summary") -> str:
    for attempt in range(MAX_RETRIES):
        try:
            if mode == "notes":
                system_msg = (
                    "You are a professional note-taking assistant. Generate detailed, "
                    "well-structured notes. Include key concepts, technical terms, "
                    "logical structure with bullet points, and relevant examples."
                )
                user_msg = f"Generate comprehensive notes from this content:\n\n{chunk}"
            else:
                system_msg = (
                    "You are an expert summary assistant. Create a concise, coherent "
                    "summary capturing essential information as a single paragraph."
                )
                user_msg = f"Generate a concise summary of this content:\n\n{chunk}"

            response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=1000,
                temperature=0.3 if mode == "summary" else 0.5
            )
            return response.choices[0].message.content

        except Exception as e:
            if "rate limit" in str(e).lower():
                wait = RATE_LIMIT_DELAY * (attempt + 1)
                log.warning("mistral.rate_limit", attempt=attempt + 1, wait=wait)
                time.sleep(wait)
            else:
                return f"Error generating {mode}: {str(e)}"

    return f"Error: Failed after {MAX_RETRIES} retries"


def generate_summary_directly(text: str) -> str:
    if not text:
        return "No content available."
    chunks = [text[i:i + TRANSCRIPT_CHUNK_SIZE_CHARS]
              for i in range(0, len(text), TRANSCRIPT_CHUNK_SIZE_CHARS)]
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as ex:
        futures = [ex.submit(summarize_chunk, c, "summary") for c in chunks]
        results = [f.result() for f in as_completed(futures)]
    combined = "\n".join(r for r in results if not r.startswith("Error"))
    return summarize_chunk(combined, "summary") if len(results) > 1 else combined


def generate_notes_directly(text: str) -> str:
    if not text:
        return "No content available."
    chunks = [text[i:i + TRANSCRIPT_CHUNK_SIZE_CHARS]
              for i in range(0, len(text), TRANSCRIPT_CHUNK_SIZE_CHARS)]
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as ex:
        futures = [ex.submit(summarize_chunk, c, "notes") for c in chunks]
        results = [f.result() for f in as_completed(futures)]
    combined = "\n\n---\n\n".join(results)
    return combined.replace("*", "").replace("#", "")


# ─────────────────────────────────────────────────────────────────────────────
# Q&A  (Qdrant + Gemini + optional Serper)
# ─────────────────────────────────────────────────────────────────────────────

def get_conversational_chain():
    prompt = PromptTemplate(
        template=(
            "You are an AI assistant helping users find relevant information "
            "from a video transcript.\n"
            "Context: {context}\nQuestion: {question}\nAnswer:"
        ),
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm_genai, chain_type="stuff", prompt=prompt)


def search_query_with_qdrant(video_id: str, query: str) -> str | None:
    results = search_qdrant(video_id, query, top_k=3)
    if not results:
        return None

    best_score = results[0]["score"]
    if best_score < SIMILARITY_THRESHOLD:
        log.info("qdrant.search.below_threshold", score=best_score)
        return None

    docs = [Document(page_content=r["text"]) for r in results]
    chain = get_conversational_chain()
    return chain.run(input_documents=docs, question=query)


def search_query_text_fallback(chunks: list, query: str) -> str:
    keywords = [w for w in query.lower().split() if len(w) > 3]
    scored = []
    for chunk in chunks:
        text_lower = chunk.get("Text", "").lower()
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scored.append((score, chunk.get("Text", "")))

    if not scored:
        return "I couldn't find relevant information in the transcript."

    scored.sort(reverse=True)
    context = "\n".join(t for _, t in scored[:2])

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": "Answer the question based on the context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)[:100]}"


def refine_answer_with_serper(query: str, context_answer: str,
                               yt_channel: str, yt_title: str) -> str:
    try:
        agent = Agent(
            role="Answer Refinement Agent",
            goal="Enhance the response using latest web data, staying aligned with the video context.",
            backstory="Verifies and refines responses using real-time search when needed.",
            verbose=False,
            memory=True,
            tools=[serper_tool],
            llm=llm,
            allow_delegation=False
        )
        task = Task(
            description=(
                f"Improve the response for query: {query}\n"
                f"Context answer: {context_answer}\n"
                f"YouTube channel: {yt_channel}, title: {yt_title}\n"
                "Give a straight-forward answer without unnecessary information."
            ),
            expected_output="A verified, refined response.",
            tools=[serper_tool],
            agent=agent
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False, process=Process.sequential)
        result = crew.kickoff(inputs={
            'query': query, 'context_answer': context_answer,
            'yt_channel': yt_channel, 'yt_title': yt_title
        })
        return result.raw
    except Exception as e:
        log.error("serper.refine.failed", error=str(e))
        return context_answer


# ─────────────────────────────────────────────────────────────────────────────
# YouTube metadata
# ─────────────────────────────────────────────────────────────────────────────

async def get_yt_details(video_id: str):
    cached = await get_cached_yt_metadata(video_id)
    if cached:
        return cached["channel"], cached["title"]

    try:
        with yt_dlp.YoutubeDL({}) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
        channel = info.get("uploader", "Unknown Channel")
        title = info.get("title", "Unknown Title")
        await cache_yt_metadata(video_id, channel, title)
        return channel, title
    except Exception as e:
        log.error("yt_details.failed", video_id=video_id, error=str(e))
        return "Unknown Channel", "Unknown Title"


# ─────────────────────────────────────────────────────────────────────────────
# Precompute  (transcript + Qdrant embeddings)
# ─────────────────────────────────────────────────────────────────────────────

async def precompute(video_id: str) -> dict:
    if is_video_indexed(video_id):
        log.info("precompute.cached", video_id=video_id)
        return {"status": "cached", "video_id": video_id}

    try:
        transcript_data = await get_or_fetch_transcript(video_id)
        chunks = chunk_transcript(transcript_data)

        success = await asyncio.to_thread(store_embeddings_qdrant, video_id, chunks)
        if not success:
            log.warning("precompute.embedding_failed", video_id=video_id)
            return {"status": "success_with_fallback", "video_id": video_id,
                    "message": "Embeddings failed; text search available."}

        await get_yt_details(video_id)  # warm metadata cache
        log.info("precompute.done", video_id=video_id, chunks=len(chunks))
        return {"status": "success", "video_id": video_id}
    except Exception as e:
        log.error("precompute.error", video_id=video_id, error=str(e))
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Audio generation
# ─────────────────────────────────────────────────────────────────────────────

async def generate_audio_and_save(text: str, lang: str, output_path: str) -> bool:
    try:
        lang_code = VOICE_CONFIGS.get(lang, 'en')

        def _gen():
            from gtts import gTTS
            gTTS(text=text, lang=lang_code, slow=False).save(output_path)

        await asyncio.to_thread(_gen)
        return True
    except Exception as e:
        log.error("audio.generate.failed", error=str(e))
        return False


async def generate_single_segment_audio(
    video_id: str, segment: dict,
    source_lang: str, target_lang: str
) -> bool:
    seg_no = segment["Segment"]
    path = os.path.join(AUDIO_DATA_DIR, video_id, f"segment_{seg_no:04d}.mp3")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return True

    try:
        # Get full English transcript for context
        english_text = await get_cached_english_transcript(video_id) or segment["Text"]
        translated = await translate_segment(
            english_text, segment["Text"], source_lang, target_lang
        )
        return await asyncio.wait_for(
            generate_audio_and_save(translated, target_lang, path),
            timeout=AUDIO_SEGMENT_TIMEOUT
        )
    except asyncio.TimeoutError:
        log.warning("audio.segment.timeout", segment=seg_no)
        return False
    except Exception as e:
        log.error("audio.segment.error", segment=seg_no, error=str(e))
        return False


async def warm_neighbor_segments(
    video_id: str, segments: list,
    segment_number: int, source_lang: str, target_lang: str
):
    """Background task: pre-generate ±5 neighbor segments."""
    idx = next((i for i, s in enumerate(segments) if s["Segment"] == segment_number), None)
    if idx is None:
        return

    neighbors = segments[max(0, idx - 2): idx] + segments[idx + 1: idx + 6]
    for seg in neighbors:
        await generate_single_segment_audio(video_id, seg, source_lang, target_lang)


# ─────────────────────────────────────────────────────────────────────────────
# Quart app
# ─────────────────────────────────────────────────────────────────────────────

app = Quart(__name__)
app = cors(app)


@app.route("/")
async def home():
    return jsonify({"status": "ok", "service": "AgentPlay API"})


@app.route("/health")
async def health():
    """Health check — returns status of all downstream services."""
    redis_ok = await ping_redis()
    qdrant_ok = await asyncio.to_thread(ping_qdrant)
    status = "healthy" if (redis_ok and qdrant_ok) else "degraded"
    return jsonify({
        "status": status,
        "redis": "ok" if redis_ok else "down",
        "qdrant": "ok" if qdrant_ok else "down"
    }), 200 if status == "healthy" else 503


@app.route("/show_transcript/<video_id>")
async def show_transcript(video_id: str):
    try:
        data = await get_or_fetch_transcript(video_id)
        return jsonify({"transcript": data["transcript"]}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        log.error("show_transcript.error", video_id=video_id, error=str(e))
        return jsonify({"error": "Failed to retrieve transcript"}), 500


@app.route("/show_data/<video_id>")
async def show_data(video_id: str):
    try:
        data = await get_or_fetch_transcript(video_id)
        return jsonify({
            "video_id": video_id,
            "transcript_exists": True,
            "source_language": data.get("source_language"),
            "total_segments": len(data["transcript"]),
            "is_indexed": is_video_indexed(video_id)
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/precompute/<video_id>")
async def precompute_route(video_id: str):
    result = await precompute(video_id)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result), 200


@app.route("/concise_summary/<video_id>")
async def concise_summary_api(video_id: str):
    try:
        cached = await get_cached_summary(video_id)
        if cached:
            return jsonify({"concise_summary": cached, "source": "cache"}), 200

        data = await get_or_fetch_transcript(video_id)
        english_text = data.get("english_text", "")
        summary = await asyncio.to_thread(generate_summary_directly, english_text)
        await cache_summary(video_id, summary)
        return jsonify({"concise_summary": summary, "source": "generated"}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        log.error("summary.error", video_id=video_id, error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/notes/<video_id>")
async def notes_api(video_id: str):
    try:
        cached = await get_cached_notes(video_id)
        if cached:
            return jsonify({"notes": cached, "source": "cache"}), 200

        data = await get_or_fetch_transcript(video_id)
        english_text = data.get("english_text", "")
        notes = await asyncio.to_thread(generate_notes_directly, english_text)
        await cache_notes(video_id, notes)
        return jsonify({"notes": notes, "source": "generated"}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        log.error("notes.error", video_id=video_id, error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/process", methods=["POST"])
async def process():
    data = await request.json
    query = data.get("query")
    video_id = data.get("video_id")
    addition_mode = data.get("addition_mode", False)

    if not video_id or not query:
        return jsonify({"error": "Missing video_id or query"}), 400

    # Ensure precomputed
    if not is_video_indexed(video_id):
        result = await precompute(video_id)
        if "error" in result:
            return jsonify(result), 400

    try:
        transcript_data = await get_or_fetch_transcript(video_id)
        chunks = chunk_transcript(transcript_data)
        yt_channel, yt_title = await get_yt_details(video_id)

        # Try Qdrant first, fallback to keyword search
        answer = search_query_with_qdrant(video_id, query)
        if answer is None:
            log.info("qa.fallback.text_search", video_id=video_id)
            answer = search_query_text_fallback(chunks, query)

        if addition_mode:
            answer = await asyncio.to_thread(
                refine_answer_with_serper, query, answer, yt_channel, yt_title
            )

        return jsonify({
            "final_answer": answer,
            "channel": yt_channel,
            "title": yt_title
        }), 200

    except Exception as e:
        log.error("process.error", video_id=video_id, error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/listen_audio/<video_id>/<target_language>/<segment_number>")
async def get_audio(video_id: str, target_language: str, segment_number: str):
    try:
        seg_no = int(segment_number)
        seg_path = os.path.join(AUDIO_DATA_DIR, video_id, f"segment_{seg_no:04d}.mp3")

        # Serve immediately if cached on disk
        if os.path.exists(seg_path):
            log.info("audio.serve.cached", video_id=video_id, segment=seg_no)
            return await send_file(seg_path, mimetype="audio/mpeg")

        # Fetch transcript
        transcript_data = await get_or_fetch_transcript(video_id)
        segments = transcript_data["transcript"]
        source_lang = transcript_data.get("source_language", "en")

        segment = next((s for s in segments if s["Segment"] == seg_no), None)
        if not segment:
            return jsonify({"error": f"Segment {seg_no} not found"}), 404

        # Generate this segment
        success = await asyncio.wait_for(
            generate_single_segment_audio(video_id, segment, source_lang, target_language),
            timeout=AUDIO_SEGMENT_TIMEOUT
        )

        if success and os.path.exists(seg_path):
            # Warm neighbors in background
            asyncio.create_task(
                warm_neighbor_segments(video_id, segments, seg_no, source_lang, target_language)
            )
            return await send_file(seg_path, mimetype="audio/mpeg")

        return jsonify({
            "status": "processing",
            "message": "Audio is being generated. Retry shortly.",
            "retry_url": f"/listen_audio/{video_id}/{target_language}/{seg_no}"
        }), 202

    except asyncio.TimeoutError:
        return jsonify({
            "status": "processing",
            "message": "Generation timed out. Retry shortly.",
            "retry_url": f"/listen_audio/{video_id}/{target_language}/{segment_number}"
        }), 202
    except Exception as e:
        log.error("audio.error", video_id=video_id, error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/cache_status")
async def cache_status():
    from backend.config.cache import get_redis
    r = await get_redis()
    keys = await r.keys("transcript:*")
    video_ids = [k.replace("transcript:", "") for k in keys]
    return jsonify({
        "cached_videos": video_ids,
        "total": len(video_ids)
    }), 200


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG)