from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import asyncio
import aiohttp
from google.cloud import secretmanager
from google.oauth2 import service_account
from google.cloud import vision, videointelligence
from enum import Enum
import re
import hashlib
import json
from datetime import datetime, timedelta
import logging
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Content Moderation API", version="1.0.0")

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"

class ContentStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"

class MediaInput(BaseModel):
    url: HttpUrl
    type: MediaType
    id: Optional[str] = None

class BatchModerationRequest(BaseModel):
    media_list: List[MediaInput]
    strict_mode: bool = True
    cache_results: bool = True

class ModerationResult(BaseModel):
    media_id: str
    status: ContentStatus
    confidence: float
    reasons: List[str]
    processing_time_ms: int
    analysis_details: dict

class BatchModerationResponse(BaseModel):
    results: List[ModerationResult]
    total_processed: int
    total_failed: int
    total_passed: int
    processing_time_ms: int

# Initialize Google Cloud clients
vision_client = None
video_client = None

def setup_credentials_from_secret():
    """Set up credentials from Google Secret Manager"""
    try:
        # Create Secret Manager client
        sm_client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        # Get the secret
        name = f"projects/{project_id}/secrets/content-moderator-credentials/versions/latest"
        response = sm_client.access_secret_version(request={"name": name})
        
        # Parse credentials
        credentials_json = response.payload.data.decode("UTF-8")
        credentials_info = json.loads(credentials_json)
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        
        # Initialize Google Cloud clients
        global vision_client, video_client
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        video_client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)
        
        logger.info("Successfully initialized Google Cloud clients with Secret Manager credentials")
        
    except Exception as e:
        logger.error(f"Failed to setup credentials: {str(e)}")
        raise

# Multilingual offensive content patterns
OFFENSIVE_PATTERNS = {
    'en': [r'\b(?:fuck|shit|damn|bitch|cunt|nigger|faggot|retard)\b'],
    'es': [r'\b(?:puta|mierda|coño|joder|cabrón)\b'],
    'fr': [r'\b(?:putain|merde|salope|connard|enculé)\b'],
    'de': [r'\b(?:scheiße|fick|hure|arschloch|fotze)\b'],
    'it': [r'\b(?:merda|cazzo|puttana|stronzo|figa)\b'],
    'pt': [r'\b(?:merda|porra|caralho|puta|fdp)\b'],
}

# Simple in-memory cache
result_cache = {}
CACHE_DURATION = timedelta(hours=int(os.getenv('CACHE_DURATION_HOURS', '24')))

def generate_content_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

async def download_media(url: str, max_size_mb: int = None) -> bytes:
    if max_size_mb is None:
        max_size_mb = int(os.getenv('MAX_MEDIA_SIZE_MB', '10'))
    
    async with aiohttp.ClientSession() as session:
        async with session.get(str(url)) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Cannot download media: {response.status}")
            
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Media file too large")
            
            return await response.read()

def check_text_content(text: str, strict_mode: bool = True) -> tuple[bool, List[str]]:
    reasons = []
    is_inappropriate = False
    text_lower = text.lower()
    
    for lang, patterns in OFFENSIVE_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.UNICODE)
            if matches:
                is_inappropriate = True
                reasons.append(f"Offensive language detected ({lang}): {', '.join(matches)}")
    
    return is_inappropriate, reasons

async def analyze_image(image_data: bytes, strict_mode: bool = True) -> tuple[bool, List[str], dict]:
    reasons = []
    is_inappropriate = False
    details = {}
    
    try:
        image = vision.Image(content=image_data)
        
        # Safe search detection
        safe_search_response = vision_client.safe_search_detection(image=image)
        text_response = vision_client.text_detection(image=image)
        
        safe_search = safe_search_response.safe_search_annotation
        details['safe_search'] = {
            'adult': safe_search.adult.name,
            'violence': safe_search.violence.name,
            'racy': safe_search.racy.name
        }
        
        # Check for inappropriate content - only flag POSSIBLE or higher
        threshold = vision.Likelihood.POSSIBLE if strict_mode else vision.Likelihood.LIKELY
        
        if safe_search.adult > threshold:
            is_inappropriate = True
            reasons.append(f"Adult content detected (confidence: {safe_search.adult.name})")
        
        if safe_search.violence > threshold:
            is_inappropriate = True
            reasons.append(f"Violence detected (confidence: {safe_search.violence.name})")
        
        if safe_search.racy > threshold:
            is_inappropriate = True
            reasons.append(f"Racy content detected (confidence: {safe_search.racy.name})")
        
        # Check text content
        if text_response.text_annotations:
            detected_text = text_response.text_annotations[0].description
            text_inappropriate, text_reasons = check_text_content(detected_text, strict_mode)
            if text_inappropriate:
                is_inappropriate = True
                reasons.extend(text_reasons)
            details['detected_text'] = detected_text[:200]
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
    
    return is_inappropriate, reasons, details

async def moderate_single_media(media: MediaInput, strict_mode: bool = True) -> ModerationResult:
    start_time = datetime.now()
    media_id = media.id or generate_content_hash(str(media.url))
    
    # Check cache
    cache_key = f"{media_id}_{strict_mode}"
    if cache_key in result_cache:
        cached_result, cached_time = result_cache[cache_key]
        if datetime.now() - cached_time < CACHE_DURATION:
            logger.info(f"Returning cached result for {media_id}")
            return cached_result
    
    try:
        # Download media
        media_data = await download_media(media.url)
        
        # Analyze (simplified for now - just images)
        if media.type == MediaType.IMAGE:
            is_inappropriate, reasons, details = await analyze_image(media_data, strict_mode)
        else:
            # For now, skip video analysis (can be added later)
            is_inappropriate, reasons, details = False, ["Video analysis not implemented yet"], {}
        
        confidence = min(0.95, 0.7 + (len(reasons) * 0.1)) if is_inappropriate else 0.9
        
        result = ModerationResult(
            media_id=media_id,
            status=ContentStatus.FAIL if is_inappropriate else ContentStatus.PASS,
            confidence=confidence,
            reasons=reasons if reasons else ["No inappropriate content detected"],
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            analysis_details=details
        )
        
        # Cache result
        result_cache[cache_key] = (result, datetime.now())
        return result
        
    except Exception as e:
        logger.error(f"Error processing media {media_id}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return ModerationResult(
            media_id=media_id,
            status=ContentStatus.FAIL,
            confidence=0.0,
            reasons=[f"Processing error: {str(e)}"],
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            analysis_details={"error": str(e)}
        )

@app.on_event("startup")
async def startup_event():
    """Initialize credentials on startup"""
    setup_credentials_from_secret()

@app.post("/moderate", response_model=BatchModerationResponse)
async def moderate_content(request: BatchModerationRequest):
    start_time = datetime.now()
    
    if len(request.media_list) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 media items per request")
    
    # Process all media concurrently
    tasks = [moderate_single_media(media, request.strict_mode) for media in request.media_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(
                ModerationResult(
                    media_id=request.media_list[i].id or f"error_{i}",
                    status=ContentStatus.FAIL,
                    confidence=0.0,
                    reasons=[f"Processing failed: {str(result)}"],
                    processing_time_ms=0,
                    analysis_details={"error": str(result)}
                )
            )
        else:
            processed_results.append(result)
    
    total_failed = sum(1 for r in processed_results if r.status == ContentStatus.FAIL)
    total_passed = len(processed_results) - total_failed
    
    return BatchModerationResponse(
        results=processed_results,
        total_processed=len(processed_results),
        total_failed=total_failed,
        total_passed=total_passed,
        processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
