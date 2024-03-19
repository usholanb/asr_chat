############################# SPEECH TO TEXT ##################################
STRAIGHT_TO_LANG = False
WHISPER_MODEL_NAME = 'large-v2'
############################## TEXT TO LLM ####################################
LAST_CONTEXT_SENTENCES = 10  # for context
DURATION = 10  # in seconds
################################ TRANSLATE ####################################
FROM_LANG = 'ru'
TO_LANG = 'en'
############################# TO TEXT SPEECH ##################################
"""
{'af': 'Afrikaans',
 'ar': 'Arabic',
 'bg': 'Bulgarian',
 'bn': 'Bengali',
 'bs': 'Bosnian',
 'ca': 'Catalan',
 'cs': 'Czech',
 'da': 'Danish',
 'de': 'German',
 'el': 'Greek',
 'en': 'English',
 'es': 'Spanish',
 'et': 'Estonian',
 'fi': 'Finnish',
 'fr': 'French',
 'gu': 'Gujarati',
 'hi': 'Hindi',
 'hr': 'Croatian',
 'hu': 'Hungarian',
 'id': 'Indonesian',
 'is': 'Icelandic',
 'it': 'Italian',
 'iw': 'Hebrew',
 'ja': 'Japanese',
 'jw': 'Javanese',
 'km': 'Khmer',
 'kn': 'Kannada',
 'ko': 'Korean',
 'la': 'Latin',
 'lv': 'Latvian',
 'ml': 'Malayalam',
 'mr': 'Marathi',
 'ms': 'Malay',
 'my': 'Myanmar (Burmese)',
 'ne': 'Nepali',
 'nl': 'Dutch',
 'no': 'Norwegian',
 'pl': 'Polish',
 'pt': 'Portuguese',
 'ro': 'Romanian',
 'ru': 'Russian',
 'si': 'Sinhala',
 'sk': 'Slovak',
 'sq': 'Albanian',
 'sr': 'Serbian',
 'su': 'Sundanese',
 'sv': 'Swedish',
 'sw': 'Swahili',
 'ta': 'Tamil',
 'te': 'Telugu',
 'th': 'Thai',
 'tl': 'Filipino',
 'tr': 'Turkish',
 'uk': 'Ukrainian',
 'ur': 'Urdu',
 'vi': 'Vietnamese',
 'zh': 'Chinese (Mandarin)',
 'zh-CN': 'Chinese (Simplified)',
 'zh-TW': 'Chinese (Mandarin/Taiwan)'}  # supported languages examples, you can print with gtts.lang.tts_langs()
"""
OUTPUT_SPEECH_LANG = 'en'
###############################################################################

PROMPT_INSTRUCTIONS = "Refine the text surrounded by triple backticks " \
                      "grammatically and return only the resulted text. Do " \
                      "your absolute best to keep the semantic meaning intact" \
                      "text surrounded by triple backticks: " \
                      "```{text}```"

################################ DEBUG ########################################
DEBUG = True
