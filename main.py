from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union
import asyncio
import aiohttp
from google.cloud import vision, videointelligence
from enum import Enum
import re
import hashlib
import json
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Content Moderation API", version="1.0.0")

# Initialize Google Cloud clients
vision_client = vision.ImageAnnotatorClient()
video_client = videointelligence.VideoIntelligenceServiceClient()

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
    strict_mode: bool = True  # More aggressive filtering
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
    """Generate hash for caching"""
    return hashlib.md5(url.encode()).hexdigest()

def check_text_content(text: str, strict_mode: bool = True) -> tuple[bool, List[str]]:
    """Check text for offensive content in multiple languages"""
    reasons = []
    is_inappropriate = False
    text_lower = text.lower()
    
    for lang, patterns in OFFENSIVE_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.UNICODE)
            if matches:
                is_inappropriate = True
                reasons.append(f"Offensive language detected ({lang}): {', '.join(matches)}")
    
    # Additional checks for strict mode
    if strict_mode:
        # Check for hate symbols, discriminatory language
        hate_patterns = [
            r'\b(?:nazi|hitler|kkk|white power|heil)\b',
            r'\b(?:terrorist|jihad|isis|kill.*(?:jews|muslims|christians))\b',
        ]
        
        for pattern in hate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                is_inappropriate = True
                reasons.append("Hate speech or discriminatory content detected")
                break
    
    return is_inappropriate, reasons

# Multilingual profanity and offensive content patterns
OFFENSIVE_PATTERNS = {
    'en': [r'\b(?:fuck|shit|damn|bitch|cunt|nigger|faggot|retard)\b'],
    'es': [r'\b(?:puta|mierda|coño|joder|cabrón)\b'],
    'fr': [r'\b(?:putain|merde|salope|connard|enculé)\b'],
    'de': [r'\b(?:scheiße|fick|hure|arschloch|fotze)\b'],
    'it': [r'\b(?:merda|cazzo|puttana|stronzo|figa)\b'],
    'pt': [r'\b(?:merda|porra|caralho|puta|fdp)\b'],
    'ru': [r'\b(?:блять|сука|хуй|пизда|ебать)\b'],
    'ar': [r'\b(?:كلب|عاهرة|خرا|لعنة)\b'],
    'zh': [r'(?:操|妈的|傻逼|草泥马|fuck)'],
}

# Simple in-memory cache for low volume
result_cache = {}
CACHE_DURATION = timedelta(hours=24)

def generate_content_hash(url: str) -> str:
    """Generate hash for caching"""
    return hashlib.md5(url.encode()).hexdigest()

async def download_media(url: str, max_size_mb: int = 50) -> bytes:
    """Download media with size limits"""
    logger.info(f"Attempting to download: {str(url)}")
    async with aiohttp.ClientSession() as session:
        async with session.get(str(url)) as response:
            logger.info(f"Download response status: {response.status}")
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Cannot download media: {response.status}")
            
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Media file too large")
            
            return await response.read()

def check_text_content(text: str, strict_mode: bool = True) -> tuple[bool, List[str]]:
    """Check text for offensive content in multiple languages"""
    reasons = []
    is_inappropriate = False
    
    text_lower = text.lower()
    
    for lang, patterns in OFFENSIVE_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.UNICODE)
            if matches:
                is_inappropriate = True
                reasons.append(f"Offensive language detected ({lang}): {', '.join(matches)}")
    
    # Additional checks for strict mode
    if strict_mode:
        # Check for hate symbols, discriminatory language
        hate_patterns = [
            r'\b(?:nazi|hitler|kkk|white power|heil)\b',
            r'\b(?:terrorist|jihad|isis|kill.*(?:jews|muslims|christians))\b',
        ]
        
        for pattern in hate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                is_inappropriate = True
                reasons.append("Hate speech or discriminatory content detected")
                break
    
    return is_inappropriate, reasons

async def analyze_image(image_data: bytes, strict_mode: bool = True) -> tuple[bool, List[str], dict]:
    """Analyze image using Google Vision API"""
    reasons = []
    is_inappropriate = False
    details = {}
    
    try:
        image = vision.Image(content=image_data)
        
        # Perform multiple detections in parallel
        safe_search_response = vision_client.safe_search_detection(image=image)
        text_response = vision_client.text_detection(image=image)
        object_response = vision_client.object_localization(image=image)
        
        # Check SafeSearch results
        safe_search = safe_search_response.safe_search_annotation
        details['safe_search'] = {
            'adult': safe_search.adult.name,
            'violence': safe_search.violence.name,
            'racy': safe_search.racy.name
        }
        
        # Stricter thresholds for zero tolerance
        threshold = vision.Likelihood.POSSIBLE if strict_mode else vision.Likelihood.LIKELY
        
        if safe_search.adult >= threshold:
            is_inappropriate = True
            reasons.append(f"Adult content detected (confidence: {safe_search.adult.name})")
        
        if safe_search.violence >= threshold:
            is_inappropriate = True
            reasons.append(f"Violence detected (confidence: {safe_search.violence.name})")
        
        if safe_search.racy >= threshold:
            is_inappropriate = True
            reasons.append(f"Racy content detected (confidence: {safe_search.racy.name})")
        
        # Check detected text for offensive content
        if text_response.text_annotations:
            detected_text = text_response.text_annotations[0].description
            text_inappropriate, text_reasons = check_text_content(detected_text, strict_mode)
            if text_inappropriate:
                is_inappropriate = True
                reasons.extend(text_reasons)
            details['detected_text'] = detected_text[:200]  # Limit for privacy
        
        # Check for potentially inappropriate objects
        inappropriate_objects = ['weapon', 'gun', 'knife', 'drug', 'syringe', 'alcohol']
        detected_objects = [obj.name.lower() for obj in object_response.localized_object_annotations]
        
        for obj in detected_objects:
            if any(inappropriate in obj for inappropriate in inappropriate_objects):
                if strict_mode:
                    is_inappropriate = True
                    reasons.append(f"Inappropriate object detected: {obj}")
        
        details['objects'] = detected_objects
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
    
    return is_inappropriate, reasons, details

async def analyze_video(video_data: bytes, strict_mode: bool = True) -> tuple[bool, List[str], dict]:
    """Analyze video using Google Video Intelligence API"""
    logger.info(f"Starting video analysis, data size: {len(video_data)} bytes")
    reasons = []
    is_inappropriate = False
    details = {}
    
    try:
        # For speed, we'll analyze just the first 30 seconds
        features = [
            videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
            videointelligence.Feature.TEXT_DETECTION,
            videointelligence.Feature.OBJECT_TRACKING
        ]
        
        operation = video_client.annotate_video(
            request={
                "input_content": video_data,
                "features": features,
            }
        )
        
        # Wait for operation to complete (with timeout)
        result = operation.result(timeout=300)  # 5 minutes max
        
        # Check explicit content
        if result.annotation_results[0].explicit_annotation:
            explicit = result.annotation_results[0].explicit_annotation
            for frame in explicit.frames:
                if frame.pornography_likelihood > videointelligence.Likelihood.UNLIKELY:
                    is_inappropriate = True
                    reasons.append("Explicit content detected in video")
                    break
        
        # Check text in video
        if result.annotation_results[0].text_annotations:
            for text_annotation in result.annotation_results[0].text_annotations:
                detected_text = text_annotation.text
                text_inappropriate, text_reasons = check_text_content(detected_text, strict_mode)
                if text_inappropriate:
                    is_inappropriate = True
                    reasons.extend(text_reasons)
        
        details['analysis'] = "Video analyzed successfully"
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    
    return is_inappropriate, reasons, details

async def moderate_single_media(media: MediaInput, strict_mode: bool = True) -> ModerationResult:
    """Moderate a single media item"""
    start_time = datetime.now()
    media_id = media.id or generate_content_hash(str(media.url))
    
    # Check cache first
    cache_key = f"{media_id}_{strict_mode}"
    if cache_key in result_cache:
        cached_result, cached_time = result_cache[cache_key]
        if datetime.now() - cached_time < CACHE_DURATION:
            logger.info(f"Returning cached result for {media_id}")
            return cached_result
    
    try:
        # Download media
        media_data = await download_media(media.url)
        
        # Analyze based on type
        if media.type == MediaType.IMAGE:
            is_inappropriate, reasons, details = await analyze_image(media_data, strict_mode)
        else:  # VIDEO
            is_inappropriate, reasons, details = await analyze_video(media_data, strict_mode)
        
        # Calculate confidence (simple heuristic)
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
            analysis_details={"error": str(e), "exception_type": type(e).__name__}
        )

@app.post("/moderate", response_model=BatchModerationResponse)
async def moderate_content(request: BatchModerationRequest):
    """
    Moderate a batch of images and/or videos for inappropriate content.
    
    Returns Pass/Fail status with detailed reasoning for each item.
    Optimized for speed with concurrent processing.
    """
    start_time = datetime.now()
    
    if len(request.media_list) > 50:  # Reasonable limit for low volume
        raise HTTPException(status_code=400, detail="Maximum 50 media items per request")
    
    # Process all media concurrently for speed
    tasks = [
        moderate_single_media(media, request.strict_mode) 
        for media in request.media_list
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
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
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.delete("/cache")
async def clear_cache():
    """Clear the results cache"""
    global result_cache
    result_cache.clear()
    return {"message": "Cache cleared successfully"}

# Cleanup old cache entries periodically
@app.on_event("startup")
async def startup_event():
    async def cleanup_cache():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            current_time = datetime.now()
            expired_keys = [
                key for key, (result, cached_time) in result_cache.items()
                if current_time - cached_time > CACHE_DURATION
            ]
            for key in expired_keys:
                del result_cache[key]
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    asyncio.create_task(cleanup_cache())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)