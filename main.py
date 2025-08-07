from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import asyncio
import aiohttp
from google.cloud import secretmanager, storage
from google.oauth2 import service_account
from google.cloud import vision, videointelligence, language_v1
from google.api_core import exceptions as google_exceptions
from enum import Enum
import re
import hashlib
import json
from datetime import datetime, timedelta
import logging
import os
import uuid
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from config import ModerationConfig
import base64
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google Cloud clients
vision_client = None
video_client = None
storage_client = None
language_client = None

# Configuration
MAX_VIDEO_SIZE_MB = int(os.getenv('MAX_VIDEO_SIZE_MB', '50'))
MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', '10'))
BUCKET_NAME = os.getenv('CONTENT_MODERATION_BUCKET', 'content-moderation-videos')

def setup_credentials_from_secret():
    """Set up credentials from Google Secret Manager"""
    try:
        sm_client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        name = f"projects/{project_id}/secrets/content-moderator-credentials/versions/latest"
        response = sm_client.access_secret_version(request={"name": name})
        
        credentials_json = response.payload.data.decode("UTF-8")
        credentials_info = json.loads(credentials_json)
        
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        
        global vision_client, video_client, storage_client, language_client
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        video_client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)
        storage_client = storage.Client(credentials=credentials, project=project_id)
        language_client = language_v1.LanguageServiceClient(credentials=credentials)
        
        logger.info("Successfully initialized Google Cloud clients")
        
    except Exception as e:
        logger.error(f"Failed to setup credentials: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting up Content Moderation API...")
    setup_credentials_from_secret()
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        if not bucket.exists():
            bucket = storage_client.create_bucket(BUCKET_NAME, location="europe-west1")
            logger.info(f"Created bucket: {BUCKET_NAME}")
    except Exception as e:
        logger.warning(f"Bucket setup issue: {str(e)}")
    
    yield
    
    logger.info("Shutting down Content Moderation API...")

app = FastAPI(
    title="Content Moderation API", 
    version="3.0.0",
    description="Simple, effective content moderation. Just tells you if content is inappropriate or not.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"

class ContentStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"

class MediaInput(BaseModel):
    url: str
    type: MediaType
    id: Optional[str] = None

class ModerationRequest(BaseModel):
    media_list: List[MediaInput]

class ModerationResult(BaseModel):
    media_id: str
    status: ContentStatus
    confidence: float
    reasons: List[str]
    processing_time_ms: int

class ModerationResponse(BaseModel):
    results: List[ModerationResult]
    total_processed: int
    total_failed: int
    total_passed: int
    processing_time_ms: int

def generate_content_hash(url: str) -> str:
    """Generate hash for content identification"""
    return hashlib.md5(url.encode()).hexdigest()

def is_data_url(url: str) -> bool:
    """Check if URL is a data URL (base64 encoded)"""
    return url.startswith('data:')

def decode_data_url(data_url: str) -> bytes:
    """Decode a data URL to bytes"""
    try:
        header, data = data_url.split(',', 1)
        decoded_data = base64.b64decode(data)
        return decoded_data
    except Exception as e:
        raise ValueError(f"Invalid data URL format: {str(e)}")

def check_text_content(text: str) -> tuple[bool, List[str]]:
    """Comprehensive text checking for inappropriate content"""
    reasons = []
    is_inappropriate = False
    text_lower = text.lower()
    
    # Layer 1: Pattern matching for explicit harmful content
    for pattern in ModerationConfig.HARMFUL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            is_inappropriate = True
            reasons.append("Inappropriate language detected")
            break
    
    # Layer 2: Direct term matching across all categories
    for term in ModerationConfig.get_all_inappropriate_terms():
        if term in text_lower:
            is_inappropriate = True
            reasons.append(f"Inappropriate term detected: {term}")
    
    # Layer 3: Context-aware checking for phrases
    concerning_phrases = [
        "how to make", "how to get", "where to buy", "for sale",
        "selling", "dealing", "trafficking", "smuggling"
    ]
    
    # Check if concerning phrases are combined with inappropriate terms
    for phrase in concerning_phrases:
        if phrase in text_lower:
            for term in ModerationConfig.WEAPONS + ModerationConfig.DRUGS_SUBSTANCES:
                if term in text_lower:
                    is_inappropriate = True
                    reasons.append(f"Suspicious activity: '{phrase}' + '{term}'")
    
    # Layer 4: AI-powered analysis if available
    if language_client and len(text.strip()) > 10:
        try:
            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            
            # Sentiment analysis
            sentiment_response = language_client.analyze_sentiment(request={"document": document})
            sentiment_score = sentiment_response.document_sentiment.score
            
            if sentiment_score < ModerationConfig.SENTIMENT_THRESHOLD:
                is_inappropriate = True
                reasons.append("Negative sentiment detected")
            
            # Content classification
            try:
                classification_response = language_client.classify_text(request={"document": document})
                for category in classification_response.categories:
                    # Flag adult, violent, or other inappropriate categories
                    concerning_categories = ['/Adult', '/Violence', '/Weapons', '/Drugs', '/Crime']
                    for concerning in concerning_categories:
                        if concerning in category.name and category.confidence > 0.5:
                            is_inappropriate = True
                            reasons.append(f"Inappropriate category: {category.name}")
                            break
            except:
                # Classification might not work for all text
                pass
                
        except Exception as e:
            logger.warning(f"AI text analysis failed: {str(e)}")
    
    # Layer 5: Length and context analysis
    words = text_lower.split()
    if len(words) > 5:  # Only for substantial text
        # Check for multiple red flags in the same text
        red_flag_count = 0
        for term in ModerationConfig.get_all_inappropriate_terms()[:20]:  # Check top terms
            if term in text_lower:
                red_flag_count += 1
        
        if red_flag_count >= 3:
            is_inappropriate = True
            reasons.append(f"Multiple concerning terms detected ({red_flag_count} terms)")
    
    return is_inappropriate, reasons

async def download_media(url: str, max_size_mb: int) -> bytes:
    """Download media with size checking"""
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(str(url)) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=400, detail=f"Cannot download media: HTTP {response.status}")
                
                chunks = []
                downloaded_size = 0
                max_size_bytes = max_size_mb * 1024 * 1024
                
                async for chunk in response.content.iter_chunked(8192):
                    downloaded_size += len(chunk)
                    if downloaded_size > max_size_bytes:
                        raise HTTPException(status_code=413, detail=f"Media file too large: >{max_size_mb}MB")
                    chunks.append(chunk)
                
                return b''.join(chunks)
                
    except aiohttp.ClientTimeout:
        raise HTTPException(status_code=408, detail="Timeout downloading media")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download media: {str(e)}")

async def analyze_image(image_data: bytes) -> tuple[bool, List[str]]:
    """Comprehensive image analysis with multiple detection layers"""
    reasons = []
    is_inappropriate = False
    
    try:
        image = vision.Image(content=image_data)
        
        # Layer 1: Google Safe Search Detection
        safe_search_response = vision_client.safe_search_detection(image=image)
        safe_search = safe_search_response.safe_search_annotation
        
        if safe_search.adult >= ModerationConfig.VISION_THRESHOLD:
            is_inappropriate = True
            reasons.append("Adult content detected")
        
        if safe_search.violence >= ModerationConfig.VISION_THRESHOLD:
            is_inappropriate = True
            reasons.append("Violence detected")
        
        if safe_search.racy >= ModerationConfig.VISION_THRESHOLD:
            is_inappropriate = True
            reasons.append("Racy content detected")
        
        # Layer 2: Object Detection with comprehensive checking
        object_response = vision_client.object_localization(image=image)
        for obj in object_response.localized_object_annotations:
            if obj.score > ModerationConfig.OBJECT_CONFIDENCE:
                obj_name_lower = obj.name.lower()
                
                # Check against all inappropriate term categories
                for term in ModerationConfig.get_all_inappropriate_terms():
                    if term in obj_name_lower:
                        is_inappropriate = True
                        reasons.append(f"Inappropriate object detected: {obj.name}")
                        break
        
        # Layer 3: Label Detection with comprehensive scene analysis
        label_response = vision_client.label_detection(image=image)
        for label in label_response.label_annotations:
            label_desc_lower = label.description.lower()
            
            # Check all labels above threshold
            if label.score > ModerationConfig.SCENE_CONFIDENCE:
                for term in ModerationConfig.get_all_inappropriate_terms():
                    if term in label_desc_lower:
                        is_inappropriate = True
                        reasons.append(f"Inappropriate scene detected: {label.description}")
                        break
            
            # Special low-confidence check for high-risk terms
            elif label.score > 0.1:  # Very low threshold for critical terms
                high_risk_terms = ModerationConfig.WEAPONS + ModerationConfig.DRUGS_SUBSTANCES
                for term in high_risk_terms:
                    if term in label_desc_lower:
                        is_inappropriate = True
                        reasons.append(f"High-risk content detected: {label.description} (low confidence)")
                        break
        
        # Layer 4: Text in Image Detection
        text_response = vision_client.text_detection(image=image)
        if text_response.text_annotations:
            detected_text = text_response.text_annotations[0].description
            text_inappropriate, text_reasons = check_text_content(detected_text)
            if text_inappropriate:
                is_inappropriate = True
                reasons.extend(text_reasons)
        
        # Layer 5: Web Detection for known inappropriate content
        try:
            web_response = vision_client.web_detection(image=image)
            if web_response.web_detection.web_entities:
                for entity in web_response.web_detection.web_entities:
                    if entity.description and entity.score > 0.3:
                        entity_desc_lower = entity.description.lower()
                        for term in ModerationConfig.get_all_inappropriate_terms():
                            if term in entity_desc_lower:
                                is_inappropriate = True
                                reasons.append(f"Web entity match: {entity.description}")
                                break
        except Exception:
            # Web detection might fail, continue without it
            pass
        
        # Layer 6: Context analysis to reduce false positives
        if is_inappropriate and reasons:
            # Check for legitimate medical/educational context
            all_text = detected_text if text_response.text_annotations else ""
            all_labels = " ".join([label.description for label in label_response.label_annotations])
            context_text = (all_text + " " + all_labels).lower()
            
            # Look for medical context
            medical_context = any(exception in context_text for exception in ModerationConfig.MEDICAL_EXCEPTIONS)
            educational_context = any(exception in context_text for exception in ModerationConfig.EDUCATIONAL_EXCEPTIONS)
            
            if medical_context:
                reasons.append("Note: Medical context detected")
            if educational_context:
                reasons.append("Note: Educational context detected")
            
            # Don't override flagging for clearly inappropriate content
            # Just add context notes for human review
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
    
    return is_inappropriate, reasons

async def analyze_video(video_data: bytes) -> tuple[bool, List[str]]:
    """Comprehensive video analysis with multiple detection layers"""
    reasons = []
    is_inappropriate = False
    
    try:
        logger.info("Starting comprehensive video analysis...")
        
        features = [
            videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.SHOT_CHANGE_DETECTION,
            videointelligence.Feature.OBJECT_TRACKING,
        ]
        
        request = videointelligence.AnnotateVideoRequest(
            input_content=video_data,
            features=features,
        )
        
        operation = video_client.annotate_video(request=request)
        result = operation.result(timeout=300)  # 5 minute timeout
        
        annotation_results = result.annotation_results[0] if result.annotation_results else None
        if not annotation_results:
            return False, ["No analysis results available"]
        
        # Layer 1: Explicit Content Detection
        if annotation_results.explicit_annotation:
            explicit = annotation_results.explicit_annotation
            inappropriate_frames = 0
            total_frames = len(explicit.frames)
            
            for frame in explicit.frames:
                if frame.pornography_likelihood >= ModerationConfig.VIDEO_THRESHOLD:
                    inappropriate_frames += 1
            
            if inappropriate_frames > 0:
                percentage = (inappropriate_frames / total_frames) * 100 if total_frames > 0 else 0
                is_inappropriate = True
                reasons.append(f"Explicit content in {inappropriate_frames}/{total_frames} frames ({percentage:.1f}%)")
        
        # Layer 2: Comprehensive Label Analysis
        if annotation_results.segment_label_annotations:
            for label_annotation in annotation_results.segment_label_annotations:
                label = label_annotation.entity.description.lower()
                confidence = max([segment.confidence for segment in label_annotation.segments], default=0)
                
                # Check against all inappropriate categories
                if confidence > ModerationConfig.SCENE_CONFIDENCE:
                    for term in ModerationConfig.get_all_inappropriate_terms():
                        if term in label:
                            is_inappropriate = True
                            reasons.append(f"Inappropriate video content: {label}")
                            break
                
                # Special handling for high-risk terms at lower confidence
                elif confidence > 0.2:
                    high_risk_terms = ModerationConfig.WEAPONS + ModerationConfig.DRUGS_SUBSTANCES + ModerationConfig.VIOLENCE_CONTENT
                    for term in high_risk_terms:
                        if term in label:
                            is_inappropriate = True
                            reasons.append(f"High-risk video content: {label} (confidence: {confidence:.2f})")
                            break
        
        # Layer 3: Shot-level Analysis
        if annotation_results.shot_label_annotations:
            concerning_shots = 0
            total_shots = len(annotation_results.shot_label_annotations)
            
            for shot_annotation in annotation_results.shot_label_annotations:
                shot_label = shot_annotation.entity.description.lower()
                confidence = max([segment.confidence for segment in shot_annotation.segments], default=0)
                
                if confidence > 0.3:
                    for term in ModerationConfig.get_all_inappropriate_terms():
                        if term in shot_label:
                            concerning_shots += 1
                            break
            
            if concerning_shots > 0:
                shot_percentage = (concerning_shots / total_shots) * 100 if total_shots > 0 else 0
                if shot_percentage > 20:  # If more than 20% of shots are concerning
                    is_inappropriate = True
                    reasons.append(f"Multiple concerning shots: {concerning_shots}/{total_shots} ({shot_percentage:.1f}%)")
        
        # Layer 4: Object Tracking
        if hasattr(annotation_results, 'object_annotations') and annotation_results.object_annotations:
            for obj_annotation in annotation_results.object_annotations:
                obj_name = obj_annotation.entity.description.lower()
                confidence = obj_annotation.confidence if hasattr(obj_annotation, 'confidence') else 0
                
                if confidence > 0.3:
                    for term in ModerationConfig.get_all_inappropriate_terms():
                        if term in obj_name:
                            is_inappropriate = True
                            reasons.append(f"Inappropriate object tracked: {obj_name}")
                            break
        
        # Layer 5: Frame-level sampling for additional analysis
        # Note: This would require extracting frames and running image analysis
        # For now, we rely on video intelligence API results
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    
    return is_inappropriate, reasons

async def moderate_single_media(media: MediaInput) -> ModerationResult:
    """Simple media moderation - just determine if content is inappropriate"""
    start_time = datetime.now()
    media_id = media.id or generate_content_hash(str(media.url))
    
    logger.info(f"Moderating media: {media_id} ({media.type})")
    
    try:
        # Handle uploaded files (data URLs)
        if is_data_url(str(media.url)):
            try:
                media_data = decode_data_url(str(media.url))
                size_mb = len(media_data) / (1024 * 1024)
                
                max_size = MAX_VIDEO_SIZE_MB if media.type == MediaType.VIDEO else MAX_IMAGE_SIZE_MB
                if size_mb > max_size:
                    return ModerationResult(
                        media_id=media_id,
                        status=ContentStatus.FAIL,
                        confidence=0.0,
                        reasons=[f"File too large: {size_mb:.1f}MB (limit: {max_size}MB)"],
                        processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                    )
                
            except ValueError as e:
                return ModerationResult(
                    media_id=media_id,
                    status=ContentStatus.FAIL,
                    confidence=0.0,
                    reasons=[f"Invalid file data: {str(e)}"],
                    processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )
        else:
            # Download from URL
            max_size = MAX_VIDEO_SIZE_MB if media.type == MediaType.VIDEO else MAX_IMAGE_SIZE_MB
            try:
                media_data = await download_media(media.url, max_size)
            except HTTPException as e:
                return ModerationResult(
                    media_id=media_id,
                    status=ContentStatus.FAIL,
                    confidence=0.0,
                    reasons=[f"Download failed: {e.detail}"],
                    processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )
        
        # Analyze based on media type
        try:
            if media.type == MediaType.IMAGE:
                is_inappropriate, reasons = await analyze_image(media_data)
            elif media.type == MediaType.VIDEO:
                is_inappropriate, reasons = await analyze_video(media_data)
            else:
                raise ValueError(f"Unsupported media type: {media.type}")
                
        except HTTPException as e:
            return ModerationResult(
                media_id=media_id,
                status=ContentStatus.FAIL,
                confidence=0.0,
                reasons=[f"Analysis failed: {e.detail}"],
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
        
        # Simple confidence calculation
        if is_inappropriate:
            confidence = 0.8 + (len(reasons) * 0.05)  # Higher confidence with more reasons
            confidence = min(0.95, confidence)
        else:
            confidence = 0.9
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        result = ModerationResult(
            media_id=media_id,
            status=ContentStatus.FAIL if is_inappropriate else ContentStatus.PASS,
            confidence=confidence,
            reasons=reasons if reasons else ["No inappropriate content detected"],
            processing_time_ms=processing_time
        )
        
        logger.info(f"Completed moderation for {media_id}: {result.status} ({processing_time}ms)")
        return result
        
    except Exception as e:
        logger.error(f"Error processing media {media_id}: {str(e)}")
        
        return ModerationResult(
            media_id=media_id,
            status=ContentStatus.FAIL,
            confidence=0.0,
            reasons=[f"Processing error: {str(e)}"],
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
        )

@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(request: ModerationRequest):
    """
    Simple content moderation - just tells you if content is inappropriate or not.
    
    Supports images and videos (<50MB). Returns PASS or FAIL with reasons.
    """
    start_time = datetime.now()
    
    if not request.media_list:
        raise HTTPException(status_code=400, detail="media_list cannot be empty")
    
    if len(request.media_list) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 media items per request")
    
    logger.info(f"Processing batch of {len(request.media_list)} items")
    
    try:
        # Process all media
        tasks = [moderate_single_media(media) for media in request.media_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in batch item {i}: {str(result)}")
                processed_results.append(
                    ModerationResult(
                        media_id=request.media_list[i].id or f"error_{i}",
                        status=ContentStatus.FAIL,
                        confidence=0.0,
                        reasons=[f"Processing failed: {str(result)}"],
                        processing_time_ms=0
                    )
                )
            else:
                processed_results.append(result)
        
        # Calculate statistics
        total_failed = sum(1 for r in processed_results if r.status == ContentStatus.FAIL)
        total_passed = len(processed_results) - total_failed
        total_processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        logger.info(f"Batch completed: {total_passed} passed, {total_failed} failed in {total_processing_time}ms")
        
        return ModerationResponse(
            results=processed_results,
            total_processed=len(processed_results),
            total_failed=total_failed,
            total_passed=total_passed,
            processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "description": "Simple content moderation - just tells you if content is inappropriate",
        "services": {
            "vision_api": "ok" if vision_client else "not_initialized",
            "video_intelligence": "ok" if video_client else "not_initialized",
            "storage": "ok" if storage_client else "not_initialized",
            "natural_language": "ok" if language_client else "not_initialized"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
