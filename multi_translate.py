from multilingual_translation import text_to_text_generation
import requests
from bs4 import BeautifulSoup

lang_ids = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "bengali": "bn",
    "Breton": "br",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greeek": "el",
    "english": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian": "fa",
    "Fulah": "ff",
    "Finnish": "fi",
    "French": "fr",
    "Western Frisian": "fy",
    "Irish": "ga",
    "Gaelic": "gd",
    "Galician": "gl",
    "Gujarati": "gu",
    "Hausa": "ha",
    "Hebrew": "he",
    "hindi": "hi",
    "Croatian": "hr",
    "Haitian": "ht",
    "Hungarian": "hu",
    "Armenian": "hy",
    "Indonesian": "id",
    "Igbo": "ig",
    "Iloko": "ilo",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Central Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Luxembourgish": "lb",
    "Ganda": "lg",
    "Lingala": "ln",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Malagasy": "mg",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Burmese": "my",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian": "no",
    "Northern Sotho": "ns",
    "Occitan": "oc",
    "Oriya": "or",
    "Panjabi": "pa",
    "Polish": "pl",
    "Pushto": "ps",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swati": "ss",
    "Sundanese": "su",
    "Swedish": "sv",
    "Swahili": "sw",
    "tamil": "ta",
    "Thai": "th",
    "Tagalog": "tl",
    "Tswana": "tn",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "chinese": "zh",
    "Zulu": "zu",
}

def model_url_list():
    url_list = []
    for i in range(0, 5):
        url_list.append(f"https://huggingface.co/models?other=m2m_100&p={i}&sort=downloads")
    return url_list

def data_scraping():
    url_list = model_url_list()
    model_list = []
    for url in url_list:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        div_class = 'grid grid-cols-1 gap-5 2xl:grid-cols-2'
        div = soup.find('div', {'class': div_class})
        for a in div.find_all('a', href=True):
            model_list.append(a['href'])
    
    for i in range(len(model_list)):
        model_list[i] = model_list[i][1:]

    return model_list

lang_list = list(lang_ids.keys())
# model_list = data_scraping()

def multilingual_translate(
    prompt: str, 
    model_id: str, 
    target_lang: str
    ):
    
    return text_to_text_generation(
        prompt=prompt, 
        model_id=model_id, 
        device='cpu',
        target_lang=lang_ids[target_lang]
        )

if __name__ == "__main__":
    print(multilingual_translate(
        'hello',
        'facebook/m2m100_418M',
        'tamil'
    ))