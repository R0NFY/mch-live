"""
Post-Call Analysis Worker Service

This service polls Yandex Object Storage for new call recordings and transcripts,
analyzes them using OceanAI (personality traits) and ChatGPT (transcript insights),
and combines the results into a unified report.
"""

import os
import json
import time
import logging
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis-worker")

# S3 Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET = os.getenv("S3_SECRET")
S3_REGION = os.getenv("S3_REGION", "ru-central1")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # Optional: proxy URL for OpenAI API
OPENAI_RELAY_TOKEN = os.getenv("OPENAI_RELAY_TOKEN") or os.getenv("RELAY_TOKEN")  # Optional: relay token for proxy auth
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))


def _normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    """
    The OpenAI Python SDK expects `base_url` to typically include `/v1`.
    To make config less error-prone, if a URL is provided and it doesn't end with `/v1`,
    we append it.
    """
    if not base_url:
        return None
    b = base_url.strip()
    if not b:
        return None
    b = b.rstrip("/")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


OPENAI_BASE_URL = _normalize_openai_base_url(OPENAI_BASE_URL)

# OceanAI Configuration
OCEANAI_LANG = os.getenv("OCEANAI_LANG", "ru")
OCEANAI_CORPUS = os.getenv("OCEANAI_CORPUS")
if not OCEANAI_CORPUS:
    OCEANAI_CORPUS = "mupta" if OCEANAI_LANG.lower().startswith("ru") else "fi"
OCEANAI_DISK = os.getenv("OCEANAI_DISK", "googledisk")
OCEANAI_CACHE_DIR = os.getenv("OCEANAI_CACHE_DIR", "/tmp/oceanai")
OCEANAI_FORCE_ASR = os.getenv("OCEANAI_FORCE_ASR", "false").lower() in ("1", "true", "yes")
TRANSCRIPT_MAX_CHARS = int(os.getenv("TRANSCRIPT_MAX_CHARS", "12000"))

# Polling interval (seconds)
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))

# Directories in Object Storage
RECORDINGS_PREFIX = "recordings/"
TRANSCRIPTS_PREFIX = "transcripts/"
REPORTS_PREFIX = "reports/"
PROCESSED_PREFIX = "processed/"

_OCEANAI_LOCK = threading.RLock()
_OCEANAI_RUNNER = None
_OCEANAI_CONFIG = None


def get_s3_client():
    """Create and return S3 client for Yandex Object Storage."""
    return boto3.client(
        service_name='s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET,
        region_name=S3_REGION
    )


def list_unprocessed_files(s3_client, prefix: str) -> list:
    """List files in a prefix that haven't been processed yet."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        files = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            # Skip if already processed (check if report exists)
            report_key = key.replace(prefix, REPORTS_PREFIX).replace('.mp4', '.json').replace('.json', '_report.json')
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=report_key)
                # Report exists, skip
            except:
                files.append(key)
        return files
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []


def download_file(s3_client, key: str, local_path: str):
    """Download a file from S3 to local path."""
    logger.info(f"Downloading {key} to {local_path}")
    s3_client.download_file(S3_BUCKET, key, local_path)


def upload_file(s3_client, local_path: str, key: str):
    """Upload a file from local path to S3."""
    logger.info(f"Uploading {local_path} to {key}")
    s3_client.upload_file(local_path, S3_BUCKET, key)


def _get_transcript_payload(transcript: dict) -> dict:
    if not isinstance(transcript, dict):
        return {}
    payload = transcript.get("transcript")
    return payload if isinstance(payload, dict) else transcript


def _normalize_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("value") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()
    if content is None:
        return ""
    return str(content).strip()


def _safe_float(value: Optional[object]) -> Optional[float]:
    try:
        if value is None:
            return None
        num = float(value)
        if math.isnan(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _load_oceanai_models(ocean, corpus: str, disk: str, lang: str) -> None:
    ocean.load_audio_model_hc(out=False)
    ocean.load_audio_model_nn(out=False)
    ocean.load_audio_model_weights_hc(
        url=ocean.weights_for_big5_["audio"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_audio_model_weights_nn(
        url=ocean.weights_for_big5_["audio"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_video_model_hc(lang=lang, out=False)
    ocean.load_video_model_deep_fe(out=False)
    ocean.load_video_model_nn(out=False)
    ocean.load_video_model_weights_hc(
        url=ocean.weights_for_big5_["video"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_video_model_weights_deep_fe(
        url=ocean.weights_for_big5_["video"][corpus]["fe"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_video_model_weights_nn(
        url=ocean.weights_for_big5_["video"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_text_features(out=False)
    ocean.setup_translation_model(out=False)
    ocean.setup_bert_encoder(force_reload=False, out=False)
    ocean.load_text_model_hc(corpus=corpus, out=False)
    ocean.load_text_model_weights_hc(
        url=ocean.weights_for_big5_["text"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_text_model_nn(corpus=corpus, out=False)
    ocean.load_text_model_weights_nn(
        url=ocean.weights_for_big5_["text"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_avt_model_b5(out=False)
    ocean.load_avt_model_weights_b5(
        url=ocean.weights_for_big5_["avt"][corpus]["b5"][disk],
        force_reload=False,
        out=False,
    )


def _get_oceanai_runner():
    global _OCEANAI_RUNNER, _OCEANAI_CONFIG
    config = (OCEANAI_LANG, OCEANAI_CORPUS, OCEANAI_DISK, OCEANAI_CACHE_DIR)
    with _OCEANAI_LOCK:
        if _OCEANAI_RUNNER is not None and _OCEANAI_CONFIG == config:
            return _OCEANAI_RUNNER

        from oceanai.modules.lab.build import Run

        os.makedirs(OCEANAI_CACHE_DIR, exist_ok=True)
        ocean = Run(lang=OCEANAI_LANG, metadata=False)
        ocean.path_to_save_ = OCEANAI_CACHE_DIR
        ocean.path_to_logs_ = os.path.join(OCEANAI_CACHE_DIR, "logs")

        _load_oceanai_models(ocean, OCEANAI_CORPUS, OCEANAI_DISK, OCEANAI_LANG)

        _OCEANAI_RUNNER = ocean
        _OCEANAI_CONFIG = config
        return _OCEANAI_RUNNER


def extract_conversation(transcript: dict) -> str:
    """
    Extract only the conversation messages from the transcript.
    Returns a clean text format: 'role: content' for each message.
    """
    conversation_lines = []

    payload = _get_transcript_payload(transcript)
    if not payload:
        return ""

    # Try to get chat_history.items first (preferred, contains clean messages)
    chat_history = payload.get('chat_history', {})
    items = chat_history.get('items', [])

    for item in items:
        # Only include actual messages, skip agent_handoff and other events
        if item.get('type') == 'message':
            role = item.get('role', 'unknown')
            content = _normalize_content(item.get('content', []))
            content = content.replace('<|im_end|>', '').strip()
            if content:
                role_display = 'Пользователь' if role == 'user' else 'Ассистент'
                conversation_lines.append(f"{role_display}: {content}")

    # Fallback: try messages list if present
    if not conversation_lines:
        messages = payload.get('messages', [])
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = _normalize_content(msg.get('content', ''))
            content = content.replace('<|im_end|>', '').strip()
            if content:
                role_display = 'Пользователь' if role == 'user' else 'Ассистент'
                conversation_lines.append(f"{role_display}: {content}")

    # Fallback: try events with user_input_transcribed type
    if not conversation_lines:
        events = payload.get('events', [])
        for event in events:
            if event.get('type') == 'user_input_transcribed':
                text = event.get('transcript', '').strip()
                if text:
                    conversation_lines.append(f"Пользователь: {text}")
            elif event.get('type') == 'conversation_item_added':
                item = event.get('item', {})
                if item.get('type') == 'message':
                    role = item.get('role', 'unknown')
                    content = _normalize_content(item.get('content', []))
                    content = content.replace('<|im_end|>', '').strip()
                    if content:
                        role_display = 'Пользователь' if role == 'user' else 'Ассистент'
                        conversation_lines.append(f"{role_display}: {content}")

    return '\n'.join(conversation_lines)


def analyze_with_oceanai(video_path: str, transcript: Optional[dict] = None) -> dict:
    """
    Analyze video with OceanAI to get Big Five personality traits.
    
    Returns dict with OCEAN scores (0-1 scale):
    - O: Openness
    - C: Conscientiousness  
    - E: Extraversion
    - A: Agreeableness
    - N: Non-Neuroticism (inverse of Neuroticism)
    """
    transcript_text = ""
    if transcript:
        transcript_text = extract_conversation(transcript)

    text_path = None
    use_asr = OCEANAI_FORCE_ASR or not transcript_text

    try:
        with _OCEANAI_LOCK:
            ocean = _get_oceanai_runner()

            ocean.path_to_dataset_ = str(Path(video_path).parent)
            ocean.ext_ = [Path(video_path).suffix.lower()]

            if transcript_text:
                text_path = Path(video_path).with_suffix(".txt")
                text_path.write_text(transcript_text, encoding="utf-8")

            logger.info("Running OceanAI analysis...")
            ok = ocean.get_avt_predictions_gradio(
                paths=[video_path],
                lang=OCEANAI_LANG,
                asr=use_asr,
                accuracy=False,
                logs=False,
                out=False,
                runtime=False,
            )

        if not ok:
            return {'success': False, 'error': 'OceanAI predictions failed', 'scores': None}

        df = ocean.df_files_
        if df is None or df.empty:
            return {'success': False, 'error': 'OceanAI produced no scores', 'scores': None}

        row = df.iloc[0].to_dict()
        scores = {
            'openness': _safe_float(row.get('Openness')),
            'conscientiousness': _safe_float(row.get('Conscientiousness')),
            'extraversion': _safe_float(row.get('Extraversion')),
            'agreeableness': _safe_float(row.get('Agreeableness')),
            'non_neuroticism': _safe_float(row.get('Non-Neuroticism')),
        }

        if not any(v is not None for v in scores.values()):
            return {'success': False, 'error': 'OceanAI produced empty scores', 'scores': None}

        logger.info(f"OceanAI analysis complete: {scores}")
        return {'success': True, 'scores': scores, 'raw_result': None}
    except Exception as e:
        logger.error(f"OceanAI analysis failed: {e}")
        return {'success': False, 'error': str(e), 'scores': None}
    finally:
        if text_path and text_path.exists():
            text_path.unlink()


def analyze_with_chatgpt(transcript: dict) -> dict:
    """
    Analyze transcript with ChatGPT to extract insights.
    
    Returns dict with:
    - summary: Brief summary of the conversation
    - topics: Key topics discussed
    - sentiment: Overall sentiment
    - recommendations: Any recommendations or action items
    """
    try:
        # In direct mode, we need OPENAI_API_KEY.
        # In relay/proxy mode, we authenticate to the relay using RELAY_TOKEN/OPENAI_RELAY_TOKEN.
        if not OPENAI_API_KEY and not OPENAI_RELAY_TOKEN:
            return {'success': False, 'error': 'OPENAI_API_KEY (or RELAY_TOKEN for proxy) not set'}
        
        api_key = OPENAI_RELAY_TOKEN if (OPENAI_BASE_URL and OPENAI_RELAY_TOKEN) else OPENAI_API_KEY
        client_kwargs = {
            "api_key": api_key,
            "timeout": OPENAI_TIMEOUT_SECONDS,
            "max_retries": OPENAI_MAX_RETRIES,
        }
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
            logger.info(f"Using OpenAI proxy: {OPENAI_BASE_URL}")
        client = OpenAI(**client_kwargs)
        
        # Format transcript for analysis
        transcript_text = extract_conversation(transcript)
        if not transcript_text:
            payload = _get_transcript_payload(transcript)
            if not payload:
                return {'success': False, 'error': 'Transcript not found'}
            transcript_text = json.dumps(payload, ensure_ascii=False, indent=2)

        if len(transcript_text) > TRANSCRIPT_MAX_CHARS:
            transcript_text = transcript_text[:TRANSCRIPT_MAX_CHARS] + "\n... [truncated]"
        
        logger.info("Sending transcript to ChatGPT for analysis...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Ты эксперт по анализу разговоров. Проанализируй транскрипт звонка и верни JSON с полями:
                    - summary: краткое резюме разговора (2-3 предложения)
                    - topics: список ключевых тем (массив строк)
                    - sentiment: общий тон разговора (positive/neutral/negative)
                    - user_intent: что хотел пользователь
                    - key_points: ключевые моменты разговора (массив строк)
                    - recommendations: рекомендации для будущего (если есть)
                    
                    Отвечай ТОЛЬКО валидным JSON без markdown."""
                },
                {
                    "role": "user",
                    "content": f"Проанализируй этот транскрипт:\n\n{transcript_text}"
                }
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            insights = json.loads(result_text)
        except json.JSONDecodeError:
            insights = {'raw_response': result_text}
        
        logger.info("ChatGPT analysis complete")
        return {
            'success': True,
            'insights': insights
        }
        
    except Exception as e:
        logger.error(f"ChatGPT analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'insights': None
        }


def combine_results(
    room_name: str,
    ocean_result: dict,
    chatgpt_result: dict,
    recording_key: str,
    transcript_key: str
) -> dict:
    """Combine OceanAI and ChatGPT results into a unified report."""
    
    report = {
        'room_name': room_name,
        'analyzed_at': datetime.now().isoformat(),
        'sources': {
            'recording': recording_key,
            'transcript': transcript_key
        },
        'personality_analysis': {
            'provider': 'OceanAI',
            'success': ocean_result.get('success', False),
            'ocean_scores': ocean_result.get('scores'),
            'error': ocean_result.get('error')
        },
        'transcript_analysis': {
            'provider': 'ChatGPT',
            'success': chatgpt_result.get('success', False),
            'insights': chatgpt_result.get('insights'),
            'error': chatgpt_result.get('error')
        }
    }
    
    # Add combined assessment if both analyses succeeded
    if ocean_result.get('success') and chatgpt_result.get('success'):
        scores = ocean_result.get('scores', {})
        report['combined_assessment'] = {
            'personality_type': determine_personality_type(scores),
            'communication_style': determine_communication_style(scores),
            'overall_success': True
        }
    
    return report


def determine_personality_type(scores: dict) -> str:
    """Determine dominant personality type from OCEAN scores."""
    if not scores:
        return 'unknown'
    
    # Find highest scoring trait
    traits = {
        'openness': scores.get('openness', 0),
        'conscientiousness': scores.get('conscientiousness', 0),
        'extraversion': scores.get('extraversion', 0),
        'agreeableness': scores.get('agreeableness', 0),
        'emotional_stability': scores.get('non_neuroticism', 0)
    }
    
    dominant = max(traits, key=traits.get)
    return dominant


def determine_communication_style(scores: dict) -> str:
    """Determine communication style based on OCEAN scores."""
    if not scores:
        return 'unknown'
    
    e = scores.get('extraversion', 0.5)
    a = scores.get('agreeableness', 0.5)
    
    if e > 0.6 and a > 0.6:
        return 'collaborative'
    elif e > 0.6 and a <= 0.6:
        return 'assertive'
    elif e <= 0.6 and a > 0.6:
        return 'supportive'
    else:
        return 'analytical'


def process_call(s3_client, recording_key: str):
    """Process a single call recording."""
    logger.info(f"Processing: {recording_key}")
    
    # Extract room name from key
    filename = Path(recording_key).stem
    room_name = filename.split('_')[0] if '_' in filename else filename
    
    # Create temp directory
    temp_dir = Path("/tmp/analysis")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Download recording
    local_video = temp_dir / Path(recording_key).name
    download_file(s3_client, recording_key, str(local_video))
    
    # Find corresponding transcript
    transcript_key = f"{TRANSCRIPTS_PREFIX}{room_name}"
    # Try to find matching transcript
    transcripts = list_unprocessed_files(s3_client, TRANSCRIPTS_PREFIX)
    matching_transcript = None
    for t in transcripts:
        if room_name in t:
            matching_transcript = t
            break
    
    transcript_data = {}
    if matching_transcript:
        local_transcript = temp_dir / Path(matching_transcript).name
        download_file(s3_client, matching_transcript, str(local_transcript))
        with open(local_transcript, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    
    # Run analyses
    ocean_result = analyze_with_oceanai(str(local_video), transcript_data)
    chatgpt_result = analyze_with_chatgpt(transcript_data)
    
    # Combine results
    report = combine_results(
        room_name=room_name,
        ocean_result=ocean_result,
        chatgpt_result=chatgpt_result,
        recording_key=recording_key,
        transcript_key=matching_transcript or "not_found"
    )
    
    # Save report locally
    report_filename = f"{filename}_report.json"
    local_report = temp_dir / report_filename
    with open(local_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Upload report to S3
    report_key = f"{REPORTS_PREFIX}{report_filename}"
    upload_file(s3_client, str(local_report), report_key)
    
    logger.info(f"Report saved to {report_key}")
    
    # Cleanup
    if local_video.exists():
        local_video.unlink()
    
    return report


def main():
    """Main worker loop."""
    logger.info("Starting Analysis Worker Service")
    logger.info(f"S3 Bucket: {S3_BUCKET}")
    logger.info(f"Poll Interval: {POLL_INTERVAL}s")
    
    s3_client = get_s3_client()
    
    while True:
        try:
            # Check for new recordings
            recordings = list_unprocessed_files(s3_client, RECORDINGS_PREFIX)
            
            if recordings:
                logger.info(f"Found {len(recordings)} unprocessed recordings")
                for recording_key in recordings:
                    try:
                        process_call(s3_client, recording_key)
                    except Exception as e:
                        logger.error(f"Error processing {recording_key}: {e}")
            else:
                logger.debug("No new recordings found")
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
