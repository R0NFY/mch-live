"""
Serverless Container Handler for Post-Call Analysis

This handler is triggered by Yandex Object Storage when a new recording
is uploaded. It processes the recording with OceanAI and ChatGPT,
then saves a combined report.

Trigger event type: create-object
Prefix filter: recordings/
"""

import os
import json
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis-handler")

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
    The OpenAI Python SDK typically expects `base_url` to include `/v1`.
    If a relay URL is provided without it, append `/v1` to reduce config mistakes.
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
OCEANAI_CACHE_DIR = os.getenv("OCEANAI_CACHE_DIR", "/app/oceanai-cache")
OCEANAI_FORCE_ASR = os.getenv("OCEANAI_FORCE_ASR", "false").lower() in ("1", "true", "yes")
TRANSCRIPT_MAX_CHARS = int(os.getenv("TRANSCRIPT_MAX_CHARS", "12000"))

# Directories in Object Storage
TRANSCRIPTS_PREFIX = os.getenv("TRANSCRIPTS_PREFIX", "transcripts/")
REPORTS_PREFIX = "reports/"

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


def download_file(s3_client, key: str, local_path: str):
    """Download a file from S3 to local path."""
    logger.info(f"Downloading {key}")
    s3_client.download_file(S3_BUCKET, key, local_path)


def upload_json(s3_client, data: dict, key: str):
    """Upload JSON data to S3."""
    logger.info(f"Uploading report to {key}")
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2),
        ContentType='application/json'
    )


def find_transcript(s3_client, room_name: str, max_retries: int = 12, retry_delay: float = 15.0) -> dict:
    """
    Find and download transcript for a given room.
    Retries with exponential backoff since transcript may be uploaded after the video.
    
    Args:
        s3_client: S3 client
        room_name: Room name to search for
        max_retries: Maximum number of retries (default: 12 = ~3 min total wait)
        retry_delay: Initial delay between retries in seconds
    """
    import time
    
    for attempt in range(max_retries):
        try:
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET, 
                Prefix=TRANSCRIPTS_PREFIX
            )
            for obj in response.get('Contents', []):
                if room_name in obj['Key']:
                    local_path = f"/tmp/{Path(obj['Key']).name}"
                    download_file(s3_client, obj['Key'], local_path)
                    with open(local_path, 'r', encoding='utf-8') as f:
                        logger.info(f"Found transcript on attempt {attempt + 1}: {obj['Key']}")
                        return json.load(f)
            
            # Transcript not found yet
            if attempt < max_retries - 1:
                wait_time = retry_delay * (1.5 ** attempt)  # Exponential backoff
                logger.info(f"Transcript not found, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"Error finding transcript (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    logger.warning(f"Transcript not found after {max_retries} attempts for room: {room_name}")
    return {}


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


def analyze_with_oceanai(video_path: str, transcript: Optional[dict] = None) -> dict:
    """
    Analyze video with OceanAI for Big Five personality traits.
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
            return {'success': False, 'error': 'OceanAI predictions failed'}

        df = ocean.df_files_
        if df is None or df.empty:
            return {'success': False, 'error': 'OceanAI produced no scores'}

        row = df.iloc[0].to_dict()
        scores = {
            'openness': _safe_float(row.get('Openness')),
            'conscientiousness': _safe_float(row.get('Conscientiousness')),
            'extraversion': _safe_float(row.get('Extraversion')),
            'agreeableness': _safe_float(row.get('Agreeableness')),
            'non_neuroticism': _safe_float(row.get('Non-Neuroticism')),
        }

        if not any(v is not None for v in scores.values()):
            return {'success': False, 'error': 'OceanAI produced empty scores'}

        logger.info(f"OceanAI analysis complete: {scores}")
        return {
            'success': True,
            'scores': scores,
        }
    except Exception as e:
        logger.error(f"OceanAI analysis failed: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        if text_path and text_path.exists():
            text_path.unlink()


def extract_conversation(transcript: dict) -> str:
    """
    Extract only the conversation messages from the transcript.
    Returns a clean text format: 'role: content' for each message.
    """
    conversation_lines = []

    payload = _get_transcript_payload(transcript)
    if not payload:
        return ""

    logger.info(f"Transcript keys: {list(payload.keys())}")

    # Try to get chat_history.items first (preferred, contains clean messages)
    chat_history = payload.get('chat_history', {})
    items = chat_history.get('items', [])

    logger.info(f"Found {len(items)} items in chat_history")

    for item in items:
        # Only include actual messages, skip agent_handoff and other events
        if item.get('type') == 'message':
            role = item.get('role', 'unknown')
            content = _normalize_content(item.get('content', []))
            content = content.replace('<|im_end|>', '').strip()
            if content:
                # Translate roles for clarity
                role_display = 'Пользователь' if role == 'user' else 'Ассистент'
                conversation_lines.append(f"{role_display}: {content}")

    # Fallback: try messages list if present
    if not conversation_lines:
        messages = payload.get('messages', [])
        logger.info(f"Trying messages fallback, found {len(messages)} messages")
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
        logger.info(f"Trying events fallback, found {len(events)} events")
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
    
    result = '\n'.join(conversation_lines)
    logger.info(f"Extracted {len(conversation_lines)} conversation lines, {len(result)} chars")
    return result


def analyze_with_chatgpt(transcript: dict) -> dict:
    """Analyze transcript with ChatGPT."""
    try:
        # If a relay is configured, it typically authenticates callers via a shared RELAY_TOKEN
        # in the Authorization header. The OpenAI Python SDK uses `api_key` as the Bearer token.
        # So when OPENAI_BASE_URL is set, prefer RELAY_TOKEN/OPENAI_RELAY_TOKEN as api_key.
        if not OPENAI_API_KEY and not OPENAI_RELAY_TOKEN:
            return {'success': False, 'error': 'OPENAI_API_KEY (or RELAY_TOKEN for proxy) not set'}
        
        # Create OpenAI client with optional base_url for proxy
        api_key = OPENAI_RELAY_TOKEN if (OPENAI_BASE_URL and OPENAI_RELAY_TOKEN) else OPENAI_API_KEY
        client_kwargs = {
            'api_key': api_key,
            # Make proxy/network flakiness survivable in production.
            'timeout': OPENAI_TIMEOUT_SECONDS,
            'max_retries': OPENAI_MAX_RETRIES,
        }
        if OPENAI_BASE_URL:
            client_kwargs['base_url'] = OPENAI_BASE_URL
            logger.info(f"Using OpenAI proxy: {OPENAI_BASE_URL}")
        client = OpenAI(**client_kwargs)
        
        # Extract only the conversation (not the entire JSON)
        conversation_text = extract_conversation(transcript)
        if not conversation_text:
            payload = _get_transcript_payload(transcript)
            if not payload:
                logger.warning("Transcript not found")
                return {'success': False, 'error': 'Transcript not found'}
            logger.warning("No conversation found, falling back to full transcript payload")
            conversation_text = json.dumps(payload, ensure_ascii=False, indent=2)

        if len(conversation_text) > TRANSCRIPT_MAX_CHARS:
            conversation_text = conversation_text[:TRANSCRIPT_MAX_CHARS] + "\n... [truncated]"
        
        logger.info(f"Sending {len(conversation_text)} chars to ChatGPT (conversation only)")
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": """Проанализируй диалог и верни JSON:
                    - summary: краткое резюме (2-3 предложения)
                    - topics: ключевые темы (массив)
                    - sentiment: тон (positive/neutral/negative)
                    - key_points: ключевые моменты (массив)
                    - user_intent: что хотел пользователь
                    Только JSON, без markdown."""
            },
            {"role": "user", "content": conversation_text}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        try:
            insights = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            insights = {'raw': response.choices[0].message.content}
        
        logger.info("ChatGPT complete")
        return {'success': True, 'insights': insights}
        
    except Exception as e:
        logger.error(f"ChatGPT failed: {e}")
        return {'success': False, 'error': str(e)}


def handler(event, context):
    """
    Main handler for Yandex Serverless Container.
    Triggered by Object Storage when new file is uploaded to recordings/
    """
    logger.info("Handler triggered")
    logger.info(f"Event: {json.dumps(event, default=str)}")
    
    try:
        # Parse event from Object Storage trigger
        messages = event.get('messages', [])
        if not messages:
            return {'statusCode': 400, 'body': 'No messages in event'}
        
        # Get object details
        details = messages[0].get('details', {})
        bucket_id = details.get('bucket_id')
        object_key = details.get('object_id')
        
        if not object_key:
            return {'statusCode': 400, 'body': 'No object_id in event'}
        
        logger.info(f"Processing: {object_key}")
        
        # Initialize S3 client
        s3_client = get_s3_client()
        
        # Download video
        local_video = f"/tmp/{Path(object_key).name}"
        download_file(s3_client, object_key, local_video)
        
        # Extract room name
        filename = Path(object_key).stem
        room_name = filename.split('_')[0]
        
        # Find transcript
        transcript = find_transcript(s3_client, room_name)
        
        # Run analyses
        ocean_result = analyze_with_oceanai(local_video, transcript)
        chatgpt_result = analyze_with_chatgpt(transcript)
        
        # Create report
        report = {
            'room_name': room_name,
            'analyzed_at': datetime.now().isoformat(),
            'recording': object_key,
            'personality': ocean_result,
            'transcript_analysis': chatgpt_result
        }
        
        # Upload report
        report_key = f"{REPORTS_PREFIX}{filename}_report.json"
        upload_json(s3_client, report, report_key)
        
        # Cleanup
        Path(local_video).unlink(missing_ok=True)
        
        logger.info(f"Done. Report: {report_key}")
        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'ok', 'report': report_key})
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        "messages": [{
            "details": {
                "bucket_id": S3_BUCKET,
                "object_id": "recordings/test_video.mp4"
            }
        }]
    }
    result = handler(test_event, None)
    print(result)
