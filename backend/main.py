import time
from quart import Quart, request, jsonify, send_file
from quart_cors import cors
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from mistralai import Mistral
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from mtranslate import translate
import json
import os
from gtts import gTTS
from crew_helper import count_words_and_translate, translate_segment
import asyncio
import faiss
import os
import yt_dlp
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool

# Disable OpenTelemetry tracing if it's causing issues
try:
    from opentelemetry import trace
    trace._TRACER_PROVIDER_SET_ONCE._done = True  # Prevent tracer provider override
    trace._TRACER_PROVIDER = None
    trace.set_tracer_provider(trace.NoOpTracerProvider())
except ImportError:
    pass

# Voice configurations for gTTS (language codes)
VOICE_CONFIGS = {
    'en': 'en',
    'hi': 'hi',
    'es': 'es',
    'fr': 'fr',
    'de': 'de',
    'ja': 'ja',
    'ko': 'ko',
    'zh': 'zh-CN',
    'it': 'it',
    'pt': 'pt',
    'ru': 'ru',
    'nl': 'nl',
    'tr': 'tr',
    'pl': 'pl',
    'id': 'id',
    'th': 'th',
    'vi': 'vi'
}

LANGUAGE_MAP = {
    'en': "English",
    'hi': "Hindi",
    'es': "Spanish",
    'fr': "French",
    'de': "German",
    'ja': "Japanese",
    'ko': "Korean",
    'zh': "Chinese (Mandarin)",
    'it': "Italian",
    'pt': "Portuguese (Brazilian)",
    'ru': "Russian",
    'nl': "Dutch",
    'tr': "Turkish",
    'pl': "Polish",
    'id': "Indonesian",
    'th': "Thai",
    'vi': "Vietnamese"
}

# Rate limiting settings 
RATE_LIMIT_DELAY = 3  # Delay in seconds between API requests
MAX_RETRIES = 3  # Maximum number of retries for failed requests
MAX_CONCURRENT_REQUESTS = 1  # Limit the number of concurrent requests (reduced to 1 to avoid 403 errors) 

# Initialize Mistral client
api_key = os.environ.get("MISTRAL_API_KEY") # Replace with your actual API key
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Initialize environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

# Initialize models and tools
llm = LLM(model="gemini/gemini-2.5-flash")
llm_genai = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
serper_tool = SerperDevTool()

# Constants
SIMILARITY_THRESHOLD = 0.3

# In-memory storage for both FAISS indexes and metadata
index_cache = {}  # For storing FAISS vector stores
metadata_cache = {}  # For storing transcript, channel, title

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds+1:02d}"

async def translate_text_async(text, target_language):
    """Translate text asynchronously using mtranslate."""
    try:
        await asyncio.sleep(1.0)  # Increased delay to avoid rate limiting
        translated_text = await asyncio.to_thread(translate, text, target_language)
        return translated_text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

async def process_transcript(transcript_original, whole_transcript, video_id, source_lang, target_lang, segment_no):
    segment_no = int(segment_no)
    path = f"data/{video_id}"
    os.makedirs(path, exist_ok=True)

    path = f"data/{video_id}/segment_{segment_no:04d}.mp3"
    
    segment_index = next((i for i, seg in enumerate(transcript_original) if seg['Segment'] == segment_no), None)
    if segment_index is None:
        print(f"❌ Segment {segment_no} not found.")
        return []

    start_index = max(0, segment_index - 5)
    end_index = min(len(transcript_original), segment_index + 11)
    print(f"start_index {start_index} end_index {end_index}")
    data = []
    
    whole_words = whole_transcript.split()
    total_words = len(whole_words)

    for i in range(start_index, end_index):
        print(i)
        if i < 0 or i >= len(transcript_original):
            continue

        segment = transcript_original[i]
        seg_no = segment['Segment']

        segment_path = f"data/{video_id}/segment_{seg_no:04d}.mp3"
        if os.path.exists(segment_path):
            print(f"✅ Segment {seg_no} already exists. Skipping...")
            continue  

        print(f"🔍 Processing missing segment {seg_no}...")

        segment_text = segment['Text']
        segment_words = segment_text.split()
        segment_word_count = len(segment_words)

        segment_start_word_idx = sum(len(s['Text'].split()) for s in transcript_original[:i])

        left_idx = max(0, segment_start_word_idx - 10)
        right_idx = min(total_words, segment_start_word_idx + segment_word_count + 15)

        if i == 0:
            left_idx = 0
            right_idx = min(50, total_words)
        elif i == len(transcript_original) - 1:
            left_idx = max(0, total_words - 50)
            right_idx = total_words

        context_text = " ".join(whole_words[left_idx:right_idx])

        max_retries = 3
        delay = 5  

        for attempt in range(max_retries):
            try:
                txt = await translate_segment(context_text, segment_text, source_lang, target_lang)
                break  
            except Exception as e:
                if "429" in str(e):  
                    print(f"Rate limit exceeded. Retrying in {delay} seconds (Attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(delay)
                    delay *= 2  
                else:
                    raise e  
        else:
            print(f"❌ Failed to translate segment {seg_no} after {max_retries} retries.")
            txt = segment_text  

        data.append({
            'Segment': seg_no,
            'Text': txt,
            'Start': segment['Start'],
            'End': segment['End'],
            'Duration': segment['Duration']
        })

    return data 

async def get_transcript_with_timestamps_async(video_id):
    ytt_api = YouTubeTranscriptApi()
    """Get the transcript with timestamps from a YouTube video ID."""
    try:
        transcript_list = None
        source_language = None

        try:
            transcript_list = ytt_api.fetch(video_id)
            source_language = 'en'
            #transcript_list = transcript.find_transcript('en')
            print("Using English transcript (no translation needed)")
        except Exception as e:
            print(f"English transcript not available: {str(e)}")
            try:
                transcript_list = ytt_api.fetch(video_id,languages=['hi', 'en'])
                source_language = 'hi'
                #transcript_list = transcript.find_transcript('hi')
                print("Using Hindi transcript, translating to English")
            except Exception as e:
                print(f"Hindi transcript not available: {str(e)}")
                try:
                    available_transcripts = ytt_api.list(video_id)
                    for transcript in available_transcripts:
                        transcript_list = transcript.fetch()
                        source_language = transcript.language_code
                        needs_translation = (source_language != 'en')
                        print(f"Using {source_language} transcript, {'translating to English' if needs_translation else 'no translation needed'}")
                        break
                except Exception as e:
                    print(f"Error finding any transcript: {str(e)}")
                    return None, None, None

        if not transcript_list:
            print("No transcript available")
            return None, None, None
        transcript_list = transcript_list.to_raw_data()
        data = []
        string_transcript = ""

        for i, entry in enumerate(transcript_list):
            segment_number = i + 1
            start_time = entry['start']
            duration = entry.get('duration', 0)
            calculated_end_time = start_time + duration

            if i < len(transcript_list) - 1:
                next_start_time = transcript_list[i + 1]['start']
                end_time = next_start_time if calculated_end_time > next_start_time else calculated_end_time
                adjusted_duration = end_time - start_time
            else:
                end_time = calculated_end_time
                adjusted_duration = duration

            data.append({
                "Segment": segment_number,
                "Text": entry['text'],
                "Start": format_timestamp(start_time),
                "End": format_timestamp(end_time),
                "Duration": format_timestamp(adjusted_duration)
            })
            string_transcript += entry['text']

        return data, source_language, string_transcript
    except Exception as e:
        print(f"Error retrieving transcript: {str(e)}")
        return None, None, None

async def generate_audio_and_save(text, lang, output_path):
    """Generate audio using gTTS."""
    try:
        # Get language code, default to 'en'
        lang_code = VOICE_CONFIGS.get(lang, 'en')
        
        # Use asyncio.to_thread to run synchronous gTTS in async context
        def generate_audio():
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(output_path)
        
        await asyncio.to_thread(generate_audio)
        return True
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False
    
async def create_audio_segments(transcript_data, video_id, target_language, specific_segment=None):
    """Generates and saves audio segments with timeout and error handling.
    If specific_segment is provided, only generates that segment."""
    audio_path = os.path.join('data', video_id)
    os.makedirs(audio_path, exist_ok=True)
    
    # Filter to specific segment if requested
    if specific_segment is not None:
        transcript_data = [s for s in transcript_data if s.get('Segment') == specific_segment]
        print(f"🎵 Generating audio for segment {specific_segment}...")
    else:
        print(f"🎵 Generating audio segments for {len(transcript_data)} segments...")
    
    # Process segments with semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(2)  # Max 2 concurrent audio tasks
    
    async def process_segment_with_semaphore(segment_text, segment_num):
        async with semaphore:
            try:
                translated_text = await translate_text_async(segment_text, target_language)
                audio_file = os.path.join(audio_path, f"segment_{segment_num:04d}.mp3")
                
                # Generate audio with timeout
                try:
                    success = await asyncio.wait_for(
                        generate_audio_and_save(translated_text, target_language, audio_file),
                        timeout=30  # 30 second timeout per segment
                    )
                    if success:
                        print(f"✅ Segment {segment_num}: Generated")
                except asyncio.TimeoutError:
                    print(f"⚠️  Segment {segment_num}: Timeout (skipped)")
                    
            except Exception as e:
                print(f"❌ Segment {segment_num}: Error - {str(e)[:50]}")
    
    tasks = []
    for segment in transcript_data:
        if not segment.get('Text', '').strip():
            continue
            
        tasks.append(process_segment_with_semaphore(segment['Text'], segment['Segment']))
    
    # Run tasks with a reasonable timeout
    if tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=300)  # 5 min total
            print(f"✅ Audio segments saved in: {audio_path}")
        except asyncio.TimeoutError:
            print(f"⚠️  Audio generation timed out (partial results saved in {audio_path})")

class TranscriptStore:
    def __init__(self, video_id, transcript_original, source_lang, original_string_transcript, whole_string_transcript_english):
        self.video_id = video_id

        if transcript_original is None:
            self.is_transcript_exists = False
        else:
            self.transcript_original = transcript_original
            self.original_video_lang = source_lang
            self.is_transcript_exists = True
            self.audio_generated = False
            self.audio_generated_language = None
            self.is_summary_generated = False
            self.summary = ""
            self.whole_string_transcript_original = original_string_transcript
            self.whole_string_transcript_english = whole_string_transcript_english
            self.is_summary_generated = False
            self.is_notes_generated = False
            self.summary = False
            self.notes = False

    @classmethod
    async def create(cls, video_id):
        transcript_original, source_lang, original_string_transcript = await get_transcript_with_timestamps_async(video_id)
        whole_string_transcript_english = original_string_transcript
        if source_lang != 'en':
            whole_string_transcript_english = await count_words_and_translate(original_string_transcript)
        return cls(video_id, transcript_original, source_lang, original_string_transcript, whole_string_transcript_english)

def summarize_chunk(chunk, mode="summary"):
    """
    Summarize a single chunk of text using Mistral.
    Supports both 'notes' and 'summary' generation modes.
    
    Args:
        chunk: Text content to process
        mode: Either "notes" (for detailed notes) or "summary" (for concise summary)
        
    Returns:
        Processed content based on the selected mode
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            if mode == "notes":
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional note-taking assistant. Generate detailed, "
                            "well-structured notes from the following content. Include:\n"
                            "1. Key concepts and main ideas\n"
                            "2. Important technical terms\n"
                            "3. Logical structure with bullet points\n"
                            "4. Relevant examples (when helpful)\n"
                            "Format with clear headings and bullet points."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Generate comprehensive notes from this content:\n\n{chunk}"
                    }
                ]
            else:  # summary mode
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert summary generation assistant. Create a concise, "
                            "coherent summary that captures the essential information."
                            "Focus on:"
                            " Key points and main ideas,Logical flow between conceptsPreservation of important termsOmitting redundant examplesStructure as a single cohesive paragraph"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Generate a concise summary of this content:\n\n{chunk}"
                    }
                ]
            
            response = client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3 if mode == "summary" else 0.5  # More creative for notes
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower():
                retries += 1
                print(f"Rate limit exceeded. Retrying ({retries}/{MAX_RETRIES})...")
                time.sleep(RATE_LIMIT_DELAY * retries)
            else:
                return f"Error generating {mode} for chunk: {str(e)}"
    return f"Error: Failed to process chunk after {MAX_RETRIES} retries"


def get_notes_from_summary(transcript):
    notes = ''
    chunk_size = 8000
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(summarize_chunk, chunk, "notes") for chunk in chunks]
        
        for future in as_completed(futures):
            try:
                notes += future.result() + "\n\n----------------------------------------\n\n"
            except Exception as e:
                notes += f"Error generating notes: {str(e)}\n\n"
    
    return notes.replace("*", "").replace("#", "")

def generate_summary_directly(transcript, max_concurrent_requests=4):
    if not transcript:
        return "No transcript available for summary generation"
    
    chunk_size = 8000
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        futures = [executor.submit(summarize_chunk, chunk, "summary") for chunk in chunks]
        
        chunk_summaries = []
        for future in as_completed(futures):
            try:
                chunk_summaries.append(future.result())
            except Exception as e:
                chunk_summaries.append(f"Error in summarization: {str(e)}")
    
    combined = "\n".join([s for s in chunk_summaries if not s.startswith("Error")])
    return summarize_chunk(combined, "summary") if len(chunk_summaries) > 1 else combined

def chunk_transcript(transcript_data, max_words=50):
    """Splits the transcript into chunks of approximately max_words words while preserving timestamps."""
    transcript_list = transcript_data.get("transcript", [])
    print(transcript_data)

    if not transcript_list or not isinstance(transcript_list, list):
        print("Error: Transcript data is empty or not in the expected format.")
        return []

    chunks = []
    current_chunk = []
    current_word_count = 0
    chunk_start_time = None

    for segment in transcript_list:
        if not isinstance(segment, dict) or "Text" not in segment:
            print(f"Skipping invalid segment: {segment}")
            continue

        words = segment["Text"].split()
        word_count = len(words)

        if chunk_start_time is None:
            chunk_start_time = segment["Start"]

        if current_word_count + word_count > max_words:
            chunk_text = " ".join(current_chunk)
            chunk_end_time = transcript_list[transcript_list.index(segment) - 1]["End"]
            
            chunks.append({
                "Text": chunk_text,
                "Start": chunk_start_time,
                "End": chunk_end_time
            })

            current_chunk = []
            current_word_count = 0
            chunk_start_time = segment["Start"]

        current_chunk.extend(words)
        current_word_count += word_count

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_end_time = transcript_list[-1]["End"]
        chunks.append({
            "Text": chunk_text,
            "Start": chunk_start_time,
            "End": chunk_end_time
        })

    return chunks

def store_embeddings(chunks):
    """Create FAISS index from document chunks with rate limiting & retry logic"""
    print(f"📊 Creating embeddings for {len(chunks)} chunks...")
    texts = [chunk["Text"] for chunk in chunks]
    
    # Retry logic with exponential backoff for rate limiting
    max_retries = 3
    base_wait_time = 5  # Start with 5 seconds
    
    for attempt in range(max_retries):
        try:
            print(f"🤖 Attempt {attempt + 1}/{max_retries}: Creating FAISS embeddings...")
            vector_store = FAISS.from_texts(texts, embedding_model)
            print("✅ FAISS embeddings created successfully")
            return vector_store
        
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  Rate limit hit (429/quota). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts")
                    raise Exception(f"Embedding API rate limit exceeded. {error_str}")
            else:
                print(f"❌ Embedding creation failed: {error_str}")
                raise
    
    return None

def check_query_relevance(vector_store, query):
    """Check if query is relevant to the transcript context
    With fallback for API failures"""
    try:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=1)
        if not docs_and_scores:
            return False
        best_match, best_score = docs_and_scores[0]
        similarity_score = 1 - best_score
        return similarity_score >= SIMILARITY_THRESHOLD
    except Exception as e:
        # Fallback: consider query relevant if it has content
        print(f"⚠️  Relevance check failed, assuming query is relevant")
        return len(query) > 0

def get_conversational_chain():
    prompt_template = """
    You are an AI assistant helping users find relevant information from a video transcript.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm_genai, chain_type="stuff", prompt=prompt)
    return chain

def search_query_with_llm(vector_store, chunks, query):
    """ Search for relevant chunks and use LLM to generate an answer 
    Falls back to simple text search if vector store fails """
    try:
        # Try vector search first
        docs_and_scores = vector_store.similarity_search_with_score(query, k=3)
        best_match, best_score = docs_and_scores[0]
        similarity_score = 1 - best_score

        if similarity_score < SIMILARITY_THRESHOLD:
            return None

        best_chunks = [Document(page_content=best_match.page_content)]
        qa_chain = get_conversational_chain()
        answer = qa_chain.run(input_documents=best_chunks, question=query)
        return answer
    
    except Exception as e:
        # Fallback: Simple text search without embeddings
        print(f"⚠️  Vector search failed ({str(e)[:50]}), using fallback text search...")
        return search_query_text_fallback(chunks, query)

def search_query_text_fallback(chunks, query):
    """ Fallback Q&A using simple text search + Mistral LLM (no embeddings needed) """
    try:
        # Split query into keywords
        query_lower = query.lower()
        keywords = [word for word in query_lower.split() if len(word) > 3]
        
        # Find best matching chunks using keyword matching
        matched_chunks = []
        for chunk in chunks:
            text_lower = chunk.get("Text", "").lower()
            keyword_count = sum(1 for kw in keywords if kw in text_lower)
            
            if keyword_count > 0:
                matched_chunks.append({
                    "text": chunk.get("Text", ""),
                    "score": keyword_count
                })
        
        if not matched_chunks:
            return "I couldn't find relevant information in the transcript to answer your question."
        
        # Sort by relevance score and take top 2
        matched_chunks.sort(key=lambda x: x["score"], reverse=True)
        context_text = "\n".join([c["text"] for c in matched_chunks[:2]])
        
        # Use Mistral to generate answer from context
        print(f"📝 Generating answer using Mistral (fallback)...")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question based on the provided context. Be concise and direct."
            },
            {
                "role": "user",
                "content": f"Context from video transcript:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]
        
        response = client.chat.complete(
            model=model,
            messages=messages,
            max_tokens=200
        )
        
        answer = response.choices[0].message.content
        return answer
    
    except Exception as e:
        print(f"❌ Fallback search failed: {str(e)}")
        return f"Error generating answer: {str(e)[:100]}"

def get_yt_details(video_id):
    """Fetches YouTube video title and channel."""
    try:
        ydl_opts = {}
        yt_url = f'https://www.youtube.com/watch?v={video_id}'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(yt_url, download=False)
            return info.get('uploader', 'Unknown Channel'), info.get('title', 'Unknown Title')
    except Exception as e:
        print(f"Error getting YouTube details: {str(e)}")
        return "Unknown Channel", "Unknown Title"

def refine_answer_with_serper(query, context_answer, yt_channel, yt_title):
    """Refine answer with additional web search info if needed"""
    try:
        refinement_agent = Agent(
            role="Answer Refinement Agent",
            goal=(
                "Enhance the generated response using the latest web data. If the query pertains to a YouTube channel, provide information specifically related to {yt_channel}. "
                "If the query is based on the video title, ensure the response focuses on {yt_title} as referenced in the transcript."
            ),
            backstory="This agent verifies and refines responses using real-time search when needed. If no valid source is found, it relies solely on embeddings.",
            verbose=True,
            memory=True,
            tools=[serper_tool],
            llm=llm,
            allow_delegation=False
        )

        refinement_task = Task(
            description=(
                "Improve the response by incorporating the latest web data for the query: {query} and the given context answer: {context_answer}. "
                "Since the query is related to the YouTube channel {yt_channel} and video title {yt_title}, ensure that the response remains aligned with this context. "
                "If no valid information is found online, generate the response only from embeddings and do not mention that the internet did not provide relevant details."
                "Give the straight forward answer and dont provide unnecessary information"
            ),
            expected_output="A well-verified and refined response with accurate information. If no valid online sources are found, the response should explicitly state that the answer is based solely on embeddings.",
            tools=[serper_tool],
            agent=refinement_agent,
        )

        crew = Crew(agents=[refinement_agent], tasks=[refinement_task], verbose=True, process=Process.sequential)
        result = crew.kickoff(inputs={'query': query, 'context_answer': context_answer, 'yt_channel': yt_channel, 'yt_title': yt_title})
        return result.raw
    except Exception as e:
        print(f"Error refining answer: {str(e)}")
        return context_answer

def is_processed(video_id):
    """ Check if a FAISS index exists in memory for the given video_id. """
    return video_id in index_cache

def store_metadata(video_id, transcript, yt_channel, yt_title, chunks):
    """ Stores transcript, channel, and title in metadata cache """
    metadata_cache[video_id] = {
        "transcript": transcript,
        "yt_channel": yt_channel,
        "yt_title": yt_title,
        "chunks": chunks
    }

def store_faiss_index(video_id, vector_store):
    """ Stores FAISS index in memory cache """
    try:
        print(f"🔹 Saving FAISS index in memory for video: {video_id}")
        index_cache[video_id] = vector_store
        print(f"✅ FAISS index stored successfully in memory")
        return True
    except Exception as e:
        print(f"🚨 Failed to store FAISS index: {str(e)}")
        return False

def load_faiss_index(video_id):
    """ Loads the FAISS index from memory """
    print(f"🔹 Checking for FAISS index in memory for video: {video_id}")

    if video_id not in index_cache:
        raise KeyError(f"🚨 FAISS index not found in memory for video: {video_id}")

    try:
        return index_cache[video_id]
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index from memory: {str(e)}")

async def precompute(video_id):
    """ Precompute transcript and embeddings if not already stored in memory 
    Falls back to text-only mode if embedding API fails """
    if is_processed(video_id):
        return {"status": "cached"}

    try:
        transcript = await show_transcript(video_id)
        if "error" in transcript:
            return {"error": transcript["error"]}

        chunks = chunk_transcript(transcript)
        
        # Try to create embeddings, but fall back gracefully
        vector_store = None
        embedding_failed = False
        
        try:
            print("🤖 Attempting to create embeddings...")
            vector_store = store_embeddings(chunks)
            print("✅ Embeddings created successfully")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⚠️  Embedding API quota exceeded. Using text-based search instead.")
                embedding_failed = True
                # Create dummy vector store - we'll use fallback search
                vector_store = None
            else:
                raise

        yt_channel, yt_title = get_yt_details(video_id)
        store_metadata(video_id, transcript, yt_channel, yt_title, chunks)

        if vector_store:
            if not store_faiss_index(video_id, vector_store):
                print("⚠️  Failed to store FAISS index, but continuing with text search")
        
        status = "success_with_fallback" if embedding_failed else "success"
        return {
            "status": status,
            "video_id": video_id,
            "message": "Using text-based search due to API quota limits" if embedding_failed else None
        }
    except Exception as e:
        return {"error": f"Precompute failed: {str(e)}"}

app = Quart(__name__)
app = cors(app)

@app.route('/')
async def home():
    return "Welcome to Quart!"

@app.route('/listen_audio/<video_id>/<target_language>/<segment_number>', methods=['GET'])
async def get_audio(video_id, target_language, segment_number):
    try:
        segment_number = int(segment_number)
        segment_path = f"data/{video_id}/segment_{segment_number:04d}.mp3"

        # If file already exists, send it immediately
        if os.path.exists(segment_path):
            print(f"✅ Audio segment {segment_number} already exists. Sending...")
            return await send_file(segment_path, mimetype="audio/mpeg")

        # File doesn't exist - generate it (with timeout)
        print(f"🎵 Generating audio for segment {segment_number} (async)...")
        
        transcript_key = f"{video_id}_transcript"
        
        if transcript_key not in globals():
            print(f"Creating new transcript for {video_id}...")
            globals()[transcript_key] = await TranscriptStore.create(video_id)
        
        transcript_data = globals()[transcript_key]
        
        if not transcript_data.is_transcript_exists:
            return jsonify({"error": "No transcript available"}), 400

        # Process audio with timeout
        try:
            segment = next(
                (seg for seg in transcript_data.transcript_original if seg.get('Segment') == segment_number),
                None
            )
            if not segment:
                return jsonify({"error": f"Segment {segment_number} not found"}), 404
            temp_trans = [segment]
            
            # Generate with timeout
            await asyncio.wait_for(
                create_audio_segments(temp_trans, video_id, target_language, specific_segment=segment_number),
                timeout=90  # 90 seconds timeout
            )
            
            # Check if file was created
            if os.path.exists(segment_path):
                print(f"✅ Audio generated successfully!")
                return await send_file(segment_path, mimetype="audio/mpeg")
            else:
                print("⚠️  Audio generation timed out but may still be processing...")
                return jsonify({
                    "status": "processing",
                    "message": "Audio is being generated. Please retry in a few seconds.",
                    "segment": segment_number,
                    "retry_url": f"/listen_audio/{video_id}/{target_language}/{segment_number}"
                }), 202  # Accepted
        
        except asyncio.TimeoutError:
            print(f"⚠️  Audio generation timed out for segment {segment_number}")
            # Start background processing
            asyncio.create_task(process_and_generate_audio(video_id, target_language, segment_number))
            return jsonify({
                "status": "processing",
                "message": "Audio generation started in background. Please retry shortly.",
                "segment": segment_number,
                "estimated_wait": "10-30 seconds",
                "retry_url": f"/listen_audio/{video_id}/{target_language}/{segment_number}"
            }), 202

    except Exception as e:
        print(f"❌ Error in get_audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred: {str(e)[:100]}"}), 500

@app.route('/show_transcript/<video_id>')
async def show_transcript(video_id):
   try:
        if f"{video_id}_transcript" in globals():
            if globals()[f"{video_id}_transcript"].is_transcript_exists:
                return {"transcript": globals()[f"{video_id}_transcript"].transcript_original}
            else:
                return {"error": "No transcript available for this VIDEO"}
        else: 
            globals()[f"{video_id}_transcript"] = await TranscriptStore.create(video_id)
            return await show_transcript(video_id)
   except Exception as e:
        return {"error": f"An error occurred: show_transcript"}
    
@app.route('/show_data/<video_id>')
async def show_data(video_id):
    try:
        if f"{video_id}_transcript" in globals():
            transcript_store = globals()[f"{video_id}_transcript"]
            if transcript_store.is_transcript_exists:
                return {
                    "video_id": transcript_store.video_id,
                    "transcript_exists": transcript_store.is_transcript_exists,
                    "original_language": transcript_store.original_video_lang,
                    "audio_generated": transcript_store.audio_generated,
                    "audio_generated_language": transcript_store.audio_generated_language,
                    "transcript_data": transcript_store.transcript_original,
                    "Summary_generates": transcript_store.is_summary_generated,
                    "total_segments": len(transcript_store.transcript_original) if transcript_store.transcript_original else 0
                }
            else:
                return {"error": "No transcript available for this video"}
        else:
            globals()[f"{video_id}_transcript"] = await TranscriptStore.create(video_id)
            return await show_data(video_id)
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.route('/concise_summary/<video_id>', methods=['GET'])
async def concise_summary_api(video_id):
    try:
        if f"{video_id}_transcript" in globals():
            transcript_store = globals()[f"{video_id}_transcript"]
            
            if transcript_store.is_summary_generated:
                concise_summary = generate_summary_directly(transcript_store.summary)
                return jsonify({"concise_summary": concise_summary}), 200
            else:
                if transcript_store.is_notes_generated:
                    concise_summary = generate_summary_directly(transcript_store.notes)
                    transcript_store.summary = concise_summary
                    transcript_store.is_summary_generated = True
                else:
                    trans_temp = transcript_store.whole_string_transcript_english
                    notes = get_notes_from_summary(trans_temp)
                    transcript_store.notes = notes
                    transcript_store.is_notes_generated = True
                    
                    concise_summary = generate_summary_directly(notes)
                    transcript_store.summary = concise_summary
                    transcript_store.is_summary_generated = True
                    
                    return jsonify({"concise_summary": concise_summary}), 200
        else: 
            globals()[f"{video_id}_transcript"] = await TranscriptStore.create(video_id)
            return await concise_summary_api(video_id)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
@app.route('/notes/<video_id>', methods=['GET'])
async def notes(video_id):
    try:
        if f"{video_id}_transcript" in globals():
            transcript_store = globals()[f"{video_id}_transcript"]
            
            if transcript_store.is_notes_generated:
                return jsonify({"notes": transcript_store.notes}), 200
            else:
                trans_temp = transcript_store.whole_string_transcript_english
                notes = get_notes_from_summary(trans_temp)
                transcript_store.is_notes_generated = True
                transcript_store.notes = notes
                
                return jsonify({"notes": notes}), 200
        else: 
            globals()[f"{video_id}_transcript"] = await TranscriptStore.create(video_id)
            return await concise_summary_api(video_id)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/precompute/<video_id>', methods=['GET'])
async def precompute_route(video_id):
    """ Route to precompute transcript & embeddings and store in memory. """
    precompute_result = await precompute(video_id)
    if "error" in precompute_result:
        return jsonify(precompute_result), 400

    return jsonify(precompute_result), 200

@app.route('/process', methods=['POST'])
async def process():
    """ Processes user query using stored transcript and FAISS index from memory.
    Falls back to text search if embeddings fail. """
    data = await request.json
    query = data.get('query')
    mode = data.get('addition_mode', True)
    video_id = data.get("video_id")

    if not video_id or not query:
        return jsonify({'error': 'Missing video_id or query'}), 400

    if not is_processed(video_id):
        precompute_result = await precompute(video_id)
        if "error" in precompute_result:
            return jsonify({"error": precompute_result["error"]}), 400

    try:
        # Try to get vector store, but it might be None if embeddings failed
        try:
            vector_store = load_faiss_index(video_id)
        except (KeyError, RuntimeError):
            # Vector store not available, will use fallback
            vector_store = None
    except Exception as e:
        return jsonify({"error": f"Failed to load FAISS index: {str(e)}"}), 400
    cached_data = metadata_cache.get(video_id, {})
    transcript = cached_data.get("transcript")
    yt_channel = cached_data.get("yt_channel", "Unknown Channel")
    yt_title = cached_data.get("yt_title", "Unknown Title")
    chunks = cached_data.get("chunks")

    if not transcript:
        return jsonify({"error": "Transcript not found in cache"}), 400

    # Check relevance only if vector store is available
    if vector_store:
        if not check_query_relevance(vector_store, query):
            return jsonify({"final_answer": "Query out of context."}), 200
        context_answer = search_query_with_llm(vector_store, chunks, query)
    else:
        # Use fallback text search
        print("📝 Using fallback text-based search (no embeddings available)")
        context_answer = search_query_text_fallback(chunks, query)
    
    refined_answer = refine_answer_with_serper(query, context_answer, yt_channel, yt_title) if mode else context_answer

    return jsonify({
        "final_answer": refined_answer,
        "channel": yt_channel,
        "title": yt_title
    })

@app.route('/cache_status', methods=['GET'])
def cache_status():
    """ Returns information about what's currently in the cache """
    return jsonify({
        "indexed_videos": list(index_cache.keys()),
        "videos_with_metadata": list(metadata_cache.keys())
    })

async def process_and_generate_audio(video_id, target_language, segment_number):
    try:
        transcript_data = globals()[f"{video_id}_transcript"]
        print("Inside process_transcript")
        if globals()[f"{video_id}_transcript"] and transcript_data.is_transcript_exists:
            segment = next(
                (seg for seg in transcript_data.transcript_original if seg.get('Segment') == segment_number),
                None
            )
            if not segment:
                print(f"⚠ Segment {segment_number} not found; skipping background processing.")
                return
            temp_trans = [segment]
            print("inside create_audio_segments")
            await create_audio_segments(temp_trans, video_id, target_language, specific_segment=segment_number)
            print(f"✅ Background processing completed for segment {segment_number}.")
        else:
            print(f"⚠ Transcript not found for {video_id}, skipping background processing.")
    except Exception as e:
        print(f"❌ Error in background processing: {e}")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
