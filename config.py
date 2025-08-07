# Content Moderation Configuration
# Comprehensive inappropriate content detection

from google.cloud import vision, videointelligence
from typing import Dict, List
import re

class ModerationConfig:
    """Comprehensive configuration for content moderation"""
    
    # Conservative thresholds - err on the side of caution
    VISION_THRESHOLD = vision.Likelihood.POSSIBLE  # Standard Google threshold
    VIDEO_THRESHOLD = videointelligence.Likelihood.POSSIBLE
    
    # Aggressive detection thresholds
    OBJECT_CONFIDENCE = 0.2  # Very low threshold to catch everything
    SCENE_CONFIDENCE = 0.25  # Low threshold for comprehensive detection
    SENTIMENT_THRESHOLD = -0.3  # More sensitive to negative content
    
    # Comprehensive inappropriate content categories
    WEAPONS = [
        'weapon', 'gun', 'rifle', 'pistol', 'firearm', 'shotgun', 'revolver',
        'knife', 'blade', 'sword', 'dagger', 'machete', 'cleaver',
        'bomb', 'explosive', 'grenade', 'ammunition', 'bullet',
        'bow', 'crossbow', 'spear', 'hatchet', 'axe'
    ]
    
    DRUGS_SUBSTANCES = [
        'drug', 'drugs', 'cocaine', 'heroin', 'meth', 'methamphetamine',
        'marijuana', 'cannabis', 'weed', 'crack', 'opium', 'morphine',
        'fentanyl', 'oxycodone', 'adderall', 'xanax', 'valium',
        'pill', 'pills', 'tablet', 'capsule', 'narcotic', 'opioid',
        'syringe', 'needle', 'injection', 'iv', 'intravenous',
        'substance', 'powder', 'crystal', 'liquid drug'
    ]
    
    VIOLENCE_CONTENT = [
        'violence', 'violent', 'fight', 'fighting', 'assault', 'attack',
        'murder', 'killing', 'death', 'blood', 'gore', 'torture',
        'abuse', 'beating', 'stabbing', 'shooting', 'riot',
        'terrorism', 'terrorist', 'extremism', 'hate crime'
    ]
    
    ADULT_CONTENT = [
        'pornography', 'porn', 'xxx', 'sex', 'sexual', 'nude', 'nudity',
        'naked', 'strip', 'stripper', 'prostitution', 'escort',
        'brothel', 'adult entertainment', 'erotic', 'fetish',
        'masturbation', 'orgasm', 'genital', 'breast', 'nipple'
    ]
    
    HATE_SPEECH = [
        'nazi', 'hitler', 'kkk', 'white supremacy', 'racial slur',
        'antisemitic', 'islamophobic', 'homophobic', 'transphobic',
        'genocide', 'ethnic cleansing', 'discrimination'
    ]
    
    SELF_HARM = [
        'suicide', 'self harm', 'cutting', 'self injury', 'overdose',
        'hanging', 'jumping', 'poison', 'self mutilation'
    ]
    
    # Combine all categories for comprehensive checking
    @classmethod
    def get_all_inappropriate_terms(cls):
        return (cls.WEAPONS + cls.DRUGS_SUBSTANCES + cls.VIOLENCE_CONTENT + 
                cls.ADULT_CONTENT + cls.HATE_SPEECH + cls.SELF_HARM)
    
    # Comprehensive harmful text patterns with more categories
    HARMFUL_PATTERNS = [
        # Profanity and slurs
        r'\b(?:fuck|shit|damn|bitch|cunt|whore|slut|nigger|faggot|retard|tranny)\b',
        
        # Hate speech
        r'\b(?:nazi|hitler|kkk|white\s+power|master\s+race|ethnic\s+cleansing)\b',
        
        # Violence and threats
        r'\b(?:kill|murder|rape|assault|torture|bomb|shoot|stab|beat\s+up)\b',
        
        # Drugs and substances
        r'\b(?:cocaine|heroin|meth|crack|weed|marijuana|pills|drugs|inject|snort)\b',
        
        # Sexual content
        r'\b(?:porn|xxx|sex|naked|nude|masturbate|orgasm|fuck|pussy|dick|cock)\b',
        
        # Self-harm
        r'\b(?:suicide|kill\s+myself|end\s+it\s+all|cut\s+myself|overdose)\b',
        
        # Weapons
        r'\b(?:gun|rifle|pistol|weapon|knife|bomb|explosive|ammunition)\b'
    ]
    
    # Medical/legitimate context exceptions (to reduce false positives)
    MEDICAL_EXCEPTIONS = [
        'medical', 'hospital', 'doctor', 'nurse', 'patient', 'treatment',
        'therapy', 'medication', 'prescription', 'vaccine', 'injection',
        'surgical', 'operation', 'procedure', 'clinic', 'healthcare'
    ]
    
    # Educational/news context exceptions
    EDUCATIONAL_EXCEPTIONS = [
        'news', 'report', 'documentary', 'education', 'awareness',
        'prevention', 'history', 'museum', 'academic', 'research',
        'study', 'analysis', 'journalism', 'article'
    ]
