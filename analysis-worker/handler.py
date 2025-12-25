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

from bigfive import analyze_with_oceanai, extract_conversation

# Load environment variables
load_dotenv()

# IMPORTANT: route all ML model weights/caches to a persistent "model volume".
# This must run BEFORE we read env vars like OCEANAI_CACHE_DIR below.
from model_paths import configure_model_caches  # noqa: E402

_MODELS_DIR = configure_model_caches()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis-handler")
logger.info("Model volume configured: MODELS_DIR=%s", _MODELS_DIR)

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


def _resolve_chatgpt_system_prompt(default_prompt: str) -> str:
    prompt = os.getenv("CHATGPT_SYSTEM_PROMPT")
    return prompt if prompt else default_prompt

TRANSCRIPT_MAX_CHARS = int(os.getenv("TRANSCRIPT_MAX_CHARS", "12000"))

# Directories in Object Storage
TRANSCRIPTS_PREFIX = os.getenv("TRANSCRIPTS_PREFIX", "transcripts/")
REPORTS_PREFIX = "reports/"

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
        system_prompt = _resolve_chatgpt_system_prompt(
            """Проанализируй диалог и верни JSON:
                    - summary: краткое резюме (2-3 предложения)
                    - topics: ключевые темы (массив)
                    - sentiment: тон (positive/neutral/negative)
                    - key_points: ключевые моменты (массив)
                    - user_intent: что хотел пользователь
                    Только JSON, без markdown."""
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_text},
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
