from pathlib import Path
import yaml

NEGATIVE_INF = -100000.0

# 'YOUR_KEY'
PERSPECTIVE_API_KEY = 'AIzaSyDFXVjdCHcCk7hUiZhTv1EgrOluBFME5k0'

PERSPECTIVE_API_ATTRIBUTES = {
    'TOXICITY', 'SEVERE_TOXICITY', 'INSULT', 'PROFANITY', 'IDENTITY_ATTACK'
}

PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
