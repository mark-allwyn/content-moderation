# Content Moderation API

AI-powered content moderation service with **balanced accuracy** - optimized to minimize both false positives and false negatives. Uses Google Cloud Vision AI, Video Intelligence, and Natural Language AI with multi-factor confidence scoring and context-aware thresholds.

## 🎯 Balanced Accuracy Approach

This API is specifically designed to avoid both:
- **False Positives**: Incorrectly flagging safe content as inappropriate
- **False Negatives**: Missing actual inappropriate content

### Key Accuracy Features

- **Multi-Factor Confidence Scoring**: Combines multiple AI detection methods with weighted confidence
- **Context-Aware Thresholds**: Different confidence levels based on content category
- **Graduated Detection Levels**: Clear violations flagged at lower thresholds, ambiguous content needs higher confidence
- **Category-Based Analysis**: Weapons and drugs detected differently than social gatherings
- **AI-Powered Validation**: Advanced sentiment analysis and scene understanding

## 🚀 Features

- **Multi-format Support**: Images (JPEG, PNG, GIF, WebP) and Videos (MP4, MOV, AVI)
- **Smart Size Handling**: Direct processing for small files, bucket-based processing for large videos (>50MB)
- **Advanced AI Detection**: Object detection, face detection, scene analysis, logo recognition
- **Multilingual Text Analysis**: Offensive language detection + AI sentiment analysis in multiple languages
- **Balanced Accuracy**: Optimized thresholds to minimize false positives and false negatives
- **Batch Processing**: Process up to 50 media items simultaneously
- **Caching System**: 24-hour result caching for improved performance
- **Async Processing**: Background processing for large videos with progress tracking
- **Comprehensive Logging**: Detailed processing logs and error handling

## 🛡️ Content Categories & Thresholds

### Immediately Flagged (Low Threshold)
Content flagged with high sensitivity due to clear policy violations:

#### ✅ **Weapons & Violence**
- **Weapons**: Guns, knives, explosives (50%+ confidence)
- **Graphic violence**: Blood, physical harm, fighting
- **Threat indicators**: Clear violence with minimal ambiguity

#### ✅ **Explicit Drugs**
- **Hard drugs**: Cocaine, heroin, methamphetamines
- **Drug paraphernalia**: Syringes, needles for drug use
- **Clear drug activity**: Obvious drug transactions or use

#### ✅ **Explicit Sexual Content**
- **Pornographic material**: Explicit sexual acts
- **Clear nudity**: Exposed genitalia, explicit sexual content
- **Sexual exploitation**: Content clearly intended for sexual gratification

### Context-Dependent (Medium Threshold)
Content requiring higher confidence or multiple factors for flagging:

#### ⚖️ **Alcohol & Substances**
- **Social drinking**: Alcohol in social settings (65%+ confidence needed)
- **Tobacco products**: Cigarettes, vaping (context matters)
- **Cannabis**: Legal in many regions, context-dependent

#### ⚖️ **Crowds & Gatherings**
- **Large gatherings**: 5+ people (flagged only with other concerning indicators)
- **Political events**: Protests, demonstrations (higher threshold)
- **Entertainment venues**: Bars, clubs (context matters)

#### ⚖️ **Suggestive Content**
- **Racy content**: Suggestive but not explicit imagery
- **Swimwear**: Beach/pool context vs. inappropriate context
- **Fashion content**: Lingerie in appropriate vs. inappropriate context

### High Threshold (Requires Very High Confidence)
Content requiring 85%+ confidence to avoid false positives:

#### 🎭 **Social & Entertainment**
- **Parties**: Social celebrations, gatherings
- **Concerts**: Music events, festivals
- **Sports events**: Including combat sports
- **Artistic content**: Fashion, art, cultural content

## 🔬 Multi-Factor Confidence Scoring

### Weighted Analysis System
The API combines multiple detection methods with weighted confidence scores:

- **Safe Search (40%)**: Primary Google Vision API results
- **Object Detection (25%)**: Weapons, drugs, inappropriate items  
- **Text Analysis (20%)**: OCR text with AI sentiment analysis
- **Scene Context (10%)**: Background environment analysis
- **Sentiment (5%)**: Supporting emotional context

### Decision Logic
- **>70% confidence**: Flag as inappropriate
- **>50% with multiple factors**: Flag with caution
- **>60% single high factor**: Flag based on strong single indicator
- **<50% confidence**: Pass as appropriate

## 🎯 Accuracy Optimizations

### False Positive Reduction
- **Context-aware detection**: Swimwear at beach vs. inappropriate context
- **Graduated thresholds**: Higher confidence needed for ambiguous content
- **Multiple factor validation**: Requires multiple indicators for borderline content
- **Category-specific thresholds**: Social gatherings need higher confidence than weapons

### False Negative Reduction  
- **Clear violation detection**: Lower thresholds for obvious inappropriate content
- **Multi-language text analysis**: Comprehensive offensive content detection
- **AI sentiment analysis**: Advanced text understanding beyond pattern matching
- **Comprehensive scene analysis**: Multiple detection layers for thorough coverage

## 🛡️ Advanced Detection Features

#### ✅ **Enhanced Text Analysis (AI-Powered)**
- **Sentiment analysis**: Detects highly negative emotional content (-0.5 threshold)
- **Content classification**: Identifies inappropriate categories with 60%+ confidence
- **Harmful content patterns**: Direct threats, hate speech, explicit content
- **Context-aware patterns**: Cannabis, alcohol, mental health (require validation)
- **Languages supported**: English, Spanish, French, German, Italian, Portuguese

#### ✅ **Smart Object Detection**
- **High-confidence objects**: Weapons, hard drugs (50%+ threshold)
- **Medium-confidence objects**: Alcohol, tobacco (60%+ threshold)  
- **Low-confidence objects**: Clothing, gatherings (75%+ threshold)
- **Context validation**: Multiple factors required for ambiguous items

#### ✅ **Intelligent Scene Analysis**
- **Clear violations**: Strip clubs, violence (55%+ threshold)
- **Context-dependent**: Bars, protests (65%+ threshold)
- **Social settings**: Parties, concerts (85%+ threshold)
- **Background analysis**: Supporting context for decision making

#### ✅ **Video Intelligence**
- **Frame-by-frame analysis**: Explicit content detection across video timeline
- **Shot change analysis**: Detects rapid editing (45+ shots flagged with other indicators)
- **Video text extraction**: OCR analysis of text appearing in videos
- **Segment analysis**: Scene-by-scene content categorization

## ⚖️ Why Balanced Accuracy Matters

### The Problem with Traditional Approaches

**Ultra-Strict Systems** (minimize false negatives):
- ❌ Flag legitimate content (parties, social events, art)
- ❌ Over-moderate fashion and lifestyle content  
- ❌ Create poor user experience with excessive blocking
- ❌ Require extensive manual review and appeals

**Lenient Systems** (minimize false positives):
- ❌ Miss actual inappropriate content
- ❌ Allow policy violations to slip through
- ❌ Create safety and brand risks
- ❌ Fail to protect users from harmful content

### Our Balanced Solution

✅ **Multi-Layer Analysis**: Combines multiple AI systems for comprehensive detection  
✅ **Context-Aware Decisions**: Understands the difference between a beach photo and inappropriate content  
✅ **Graduated Confidence**: Clear violations detected with high sensitivity, ambiguous content with caution  
✅ **Reduced Manual Review**: Accurate automated decisions reduce human intervention needs  
✅ **Better User Experience**: Legitimate content passes while maintaining strong safety standards

## ⚠️ Content Limitations

#### ❌ Gesture Recognition & Social Cues
- **Hand gestures**: Vulgar finger gestures, offensive hand signs
- **Body language**: Threatening poses, inappropriate gestures
- **Social appropriateness**: Context-dependent inappropriate behavior

#### ❌ Audio Content
- **No audio analysis**: Spoken words, music, sound effects are not processed
- **Voice-based threats**: Verbal abuse or inappropriate audio content
- **Background music**: Copyright or inappropriate audio tracks

#### ❌ Context-Dependent Content
- **Sarcasm/irony**: Cannot understand contextual meaning beyond AI sentiment analysis
- **Cultural nuances**: May miss culturally specific inappropriate content
- **Implied threats**: Subtle or coded threatening language

#### ❌ Advanced Deepfakes
- **Sophisticated AI-generated content**: May not detect high-quality deepfakes
- **Face swapping**: Advanced face manipulation techniques
- **Voice cloning**: AI-generated speech (not analyzed anyway)

#### ❌ Live Streaming
- **Real-time content**: Only processes static files, not live streams
- **Dynamic content**: Cannot moderate continuously changing content

#### ❌ Copyright/IP Violations
- **Copyrighted material**: Does not detect unauthorized use of copyrighted content
- **Trademark infringement**: Brand logo or trademark violations
- **Intellectual property**: Patent or trade secret violations

#### ❌ Privacy Violations
- **Personal information**: Does not detect PII, addresses, phone numbers
- **Data leaks**: Cannot identify sensitive personal data exposure
- **GDPR compliance**: Does not ensure privacy regulation compliance

#### ❌ Misinformation/Disinformation
- **Fake news**: Cannot verify factual accuracy of content
- **Medical misinformation**: Unverified health claims
- **Political manipulation**: Biased or false political content

## 🔧 Setup & Installation

### Prerequisites

- Python 3.8+
- Google Cloud Project with enabled APIs:
  - Cloud Vision API
  - Video Intelligence API
  - Natural Language API
  - Secret Manager API
  - Cloud Storage API

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd content-moderation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Enable Required APIs**
   ```bash
   gcloud services enable vision.googleapis.com
   gcloud services enable videointelligence.googleapis.com
   gcloud services enable language.googleapis.com
   gcloud services enable secretmanager.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

4. **Set up Google Cloud credentials**
   - Create a service account with appropriate permissions
   - Store credentials in Google Secret Manager as `content-moderator-credentials`
   - Set environment variables:
   ```bash
   export GOOGLE_CLOUD_PROJECT=your-project-id
   export CONTENT_MODERATION_BUCKET=your-bucket-name
   ```

4. **Configure environment variables** (optional)
   ```bash
   # Create .env file
   MAX_VIDEO_SIZE_MB=50          # Direct processing limit for videos
   MAX_IMAGE_SIZE_MB=10          # Direct processing limit for images
   CACHE_DURATION_HOURS=24       # Result cache duration
   ```

6. **Run the server**
   ```bash
   python main.py
   # OR
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## 📖 API Usage

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 Configuration & Transparency

### Understanding Confidence Levels

**Endpoint**: `GET /confidence-levels`

Get detailed explanation of Google Vision API confidence scoring and balanced thresholds:

```bash
curl "http://localhost:8000/confidence-levels"
```

Returns comprehensive information about:
- Google Vision API likelihood levels (VERY_UNLIKELY to VERY_LIKELY)
- Balanced mode thresholds for different content types
- Multi-factor scoring weights and decision logic
- Accuracy improvement strategies
- Content categorization approach

### Current Configuration

**Endpoint**: `GET /moderation-config`

View active moderation settings and thresholds:

```bash
curl "http://localhost:8000/moderation-config"
```

Returns:
- Current detection thresholds for all categories
- Confidence weights for multi-factor scoring
- Categorized object and scene lists
- Active features and optimization settings

## 📡 API Usage

### 1. Batch Content Moderation

Process multiple images and small videos (<50MB) simultaneously.

**Endpoint**: `POST /moderate`

```bash
curl -X POST "http://localhost:8000/moderate" \
  -H "Content-Type: application/json" \
  -d '{
    "media_list": [
      {
        "url": "https://example.com/image1.jpg",
        "type": "image",
        "id": "img_001"
      },
      {
        "url": "https://example.com/video1.mp4",
        "type": "video",
        "id": "vid_001"
      }
    ],
    "strict_mode": true,
    "cache_results": true
  }'
```

**Response Example**:
```json
{
  "results": [
    {
      "media_id": "img_001",
      "status": "PASS",
      "confidence": 0.9,
      "reasons": ["No inappropriate content detected"],
      "processing_time_ms": 1250,
      "analysis_details": {
        "safe_search": {
          "adult": "VERY_UNLIKELY",
          "adult_confidence": 1,
          "violence": "UNLIKELY", 
          "violence_confidence": 2,
          "racy": "POSSIBLE",
          "racy_confidence": 3,
          "spoof": "VERY_UNLIKELY",
          "spoof_confidence": 1,
          "medical": "UNLIKELY",
          "medical_confidence": 2
        },
        "detected_objects": [
          {"name": "Person", "confidence": 0.92},
          {"name": "Clothing", "confidence": 0.78}
        ],
        "faces_detected": 2,
        "top_labels": [
          {"label": "Person", "confidence": 0.95},
          {"label": "Outdoor", "confidence": 0.87}
        ],
        "scene_labels": [],
        "detected_logos": []
      }
    }
  ],
  "total_processed": 1,
  "total_failed": 0,
  "total_passed": 1,
  "processing_time_ms": 1300
}
```

### 📊 Understanding Confidence Levels

Access `GET /confidence-levels` for detailed explanation, or see this summary:

| **Level** | **Numeric** | **Range** | **Description** | **Your Threshold** |
|-----------|-------------|-----------|-----------------|-------------------|
| VERY_LIKELY | 5 | 90-100% | Almost certain | ✅ **Flagged** |
| LIKELY | 4 | 70-89% | Probably contains | ✅ **Flagged** |
| POSSIBLE | 3 | 50-69% | Might contain | ✅ **Flagged** |
| UNLIKELY | 2 | 20-49% | Probably doesn't | ✅ **Flagged** (Ultra-strict) |
| VERY_UNLIKELY | 1 | 0-19% | Almost certain safe | ✅ **Flagged** (Ultra-strict) |

⚠️ **Your ultra-strict mode flags content even at VERY_UNLIKELY level!**

### 2. Large Video Processing

For videos >50MB, use the bucket upload system.

#### Step 1: Get Upload URL

**Endpoint**: `POST /video/upload-url`

```bash
curl -X POST "http://localhost:8000/video/upload-url"
```

**Response**:
```json
{
  "upload_url": "https://storage.googleapis.com/...",
  "video_id": "uuid-here",
  "expires_in_hours": 2,
  "max_file_size_gb": 5,
  "instructions": [
    "Upload your video file using HTTP PUT to the upload_url",
    "After successful upload, call POST /video/analyze with the video_id"
  ]
}
```

#### Step 2: Upload Video

```bash
curl -X PUT "UPLOAD_URL_FROM_STEP_1" \
  -H "Content-Type: video/mp4" \
  --data-binary @your-large-video.mp4
```

#### Step 3: Start Analysis

**Endpoint**: `POST /video/analyze`

```bash
curl -X POST "http://localhost:8000/video/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "uuid-from-step-1",
    "strict_mode": true
  }'
```

#### Step 4: Check Status

**Endpoint**: `GET /video/status/{task_id}`

```bash
curl "http://localhost:8000/video/status/your-task-id"
```

## ⚙️ Configuration Options

### Ultra-Strict Mode Configuration

To enable the **strictest possible settings**, always set `strict_mode: true` in your API requests:

```json
{
  "media_list": [...],
  "strict_mode": true,
  "cache_results": true
}
```

**Ultra-Strict Mode Features:**
- **VERY_UNLIKELY threshold** for adult/violence detection (catches almost everything)
- **Object detection at 30% confidence** (vs 50% normal)
- **Groups of 3+ people flagged** as potentially risky
- **Scene analysis at 50% confidence** (vs 70% normal)
- **Negative sentiment at -0.3 score** (vs -0.6 normal)
- **Text analysis on 5+ characters** (vs 10+ normal)
- **Video shot changes >30** flagged (vs 50 normal)
- **Expanded pattern matching** for drugs, violence, mental health content
- **Additional content categories** flagged (politics, combat sports, humor)

⚠️ **Warning**: Ultra-strict mode will produce **many false positives** and may flag legitimate content.

### Ultra-Strict Mode vs Normal Mode

| Feature | Normal Mode | **Ultra-Strict Mode** |
|---------|-------------|-------------|
| **Adult Content** | POSSIBLE threshold | **VERY_UNLIKELY threshold** |
| **Violence** | POSSIBLE threshold | **VERY_UNLIKELY threshold** |
| **Object Detection** | Medium confidence (>0.5) | **Low confidence (>0.3)** |
| **Crowd Detection** | Large crowds flagged (>5 faces) | **Small groups flagged (>3 faces)** |
| **Scene Analysis** | Medium confidence (>0.7) | **Low confidence (>0.5)** |
| **Sentiment Analysis** | Strong negative only | **Mild negative emotions (-0.3)** |
| **Text Length** | 10+ characters analyzed | **5+ characters analyzed** |
| **Video Shot Changes** | >50 shots flagged | **>30 shots flagged** |
| **Content Categories** | High confidence (>0.7) | **Medium confidence (>0.5)** |
| **Pattern Matching** | Basic hate speech | **Comprehensive patterns (drugs, violence, etc.)** |
| **False Positives** | Lower chance | **Very high chance** |
| **Sensitivity** | Moderate | **Maximum** |

### File Size Limits

| Media Type | Direct Processing | Bucket Processing |
|------------|------------------|-------------------|
| **Images** | Up to 10MB | Not applicable |
| **Videos** | Up to 50MB | Up to 5GB |

### Supported Formats

| Type | Formats |
|------|---------|
| **Images** | JPEG, PNG, GIF, WebP, BMP, TIFF |
| **Videos** | MP4, MOV, AVI, MKV, WebM |

## 🔍 Monitoring & Management

### Health Check
```bash
curl "http://localhost:8000/health"
```

### Confidence Levels Explanation
```bash
curl "http://localhost:8000/confidence-levels"
```

### Cache Statistics
```bash
curl "http://localhost:8000/cache/stats"
```

### Clear Cache
```bash
# Clear all cache
curl -X DELETE "http://localhost:8000/cache"

# Clear specific media cache
curl -X DELETE "http://localhost:8000/cache/media_id_here"
```

## 🚨 Error Handling

The API provides detailed error responses with appropriate HTTP status codes:

- **400**: Bad request (invalid input, file too large)
- **403**: Access denied to media URL
- **404**: Media not found or task not found
- **408**: Timeout (download or processing timeout)
- **413**: File too large
- **429**: Rate limited or quota exceeded
- **500**: Internal server error
- **503**: Service temporarily unavailable

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Client App    │    │  FastAPI     │    │  Google Cloud   │
│                 │◄──►│  Server      │◄──►│  Vision API     │
│                 │    │              │    │  Video Intel    │
└─────────────────┘    └──────────────┘    │  Language AI    │
                              │             └─────────────────┘
                              ▼
                       ┌──────────────┐
                       │ Cloud Storage│
                       │ (Large Videos)│
                       └──────────────┘
```

## 📊 Performance Considerations

- **Images**: Typically processed in 1-3 seconds
- **Small Videos** (<50MB): 30 seconds to 3 minutes
- **Large Videos** (>50MB): 1-4 hours (background processing)
- **Batch Processing**: Concurrent processing for improved throughput
- **Caching**: 24-hour cache reduces repeated processing costs

## 🔒 Security & Privacy

- **No Data Storage**: Media files are not permanently stored
- **Automatic Cleanup**: Large videos deleted after processing
- **Secure Upload**: Signed URLs with expiration for bucket uploads
- **Error Sanitization**: Sensitive information filtered from error messages
- **Request Validation**: Input sanitization and size limits

## 📞 Support & Troubleshooting

### Common Issues

1. **"Video too large for direct processing"**
   - Use the bucket upload system for videos >50MB
   - Consider compressing the video

2. **"Natural Language AI analysis failed"**
   - Check Natural Language API quotas and billing
   - Verify API is enabled in Google Cloud Console
   - Text analysis will continue with basic pattern matching

3. **"Vision API temporarily unavailable"**
   - Check Google Cloud service status
   - Verify API quotas and billing

3. **"Vision API temporarily unavailable"**
   - Check Google Cloud service status
   - Verify API quotas and billing

4. **"Download failed: Timeout"**
   - Ensure media URLs are accessible
   - Check network connectivity

4. **"Download failed: Timeout"**
   - Ensure media URLs are accessible
   - Check network connectivity

5. **"Access denied to media URL"**
   - Verify URL permissions
   - Check for authentication requirements

### Logs

The application provides comprehensive logging:
- Request processing details
- Error messages with stack traces
- Performance metrics
- Cache hit/miss statistics

## 📄 License

MIT License

Copyright (c) 2025 Allwyn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

