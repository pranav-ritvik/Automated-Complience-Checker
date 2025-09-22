# ========================================
# Legal Metrology Compliance Checker - Flask Backend (FIXED)
# ========================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from io import BytesIO
import pycountry
import spacy
from langdetect import detect, LangDetectException
import tempfile
import os
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = "314fd97af36157ff3b8a145f7f1afe01"
SCRAPERAPI_URL = "http://api.scraperapi.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# Global dictionary to cache loaded spaCy models
loaded_models = {}

# ========================================
# UTILITY FUNCTIONS (FIXED)
# ========================================

def get_html(url, render=True, premium=True, country="in"):
    """Get HTML using ScraperAPI"""
    try:
        params = {
            "api_key": API_KEY,
            "url": url,
            "country_code": country,
            "render": "true" if render else "false",
            "keep_headers": "true",
        }
        if premium:
            params["premium"] = "true"
        
        response = requests.get(SCRAPERAPI_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching HTML for {url}: {e}")
        # Fallback to direct request
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except:
            raise Exception(f"Failed to fetch HTML from {url}")

def split_quantity(text):
    """Extract quantity and unit from text"""
    if not text:
        return None, None
    match = re.search(r"(\d+\.?\d*)\s?(g|kg|ml|l|pcs|pack|packs|tablet|capsule)", text, re.I)
    if match:
        return match.group(1), match.group(2).lower()
    return None, None

def clean_product_name(name):
    """Clean product name"""
    if not name:
        return None
    base = name.split("|")[0].strip()
    base = base.split("(")[0].strip()
    words = base.split()
    if len(words) > 5:
        base = " ".join(words[:5])
    return base

def parse_search_page(html):
    """Parse Amazon search results page"""
    soup = BeautifulSoup(html, "html.parser")
    products = []
    
    # Try multiple selectors for Amazon products
    selectors = [
        "div[data-asin]",
        "[data-component-type='s-search-result']",
        ".s-result-item"
    ]
    
    items = []
    for selector in selectors:
        items = soup.select(selector)
        if items:
            break
    
    for item in items:
        try:
            # Get ASIN
            asin = item.get("data-asin")
            if not asin:
                continue
            
            # Get title
            title_selectors = [
                "span.a-size-medium.a-color-base.a-text-normal",
                "h2 a span",
                "span.a-text-normal",
                "h2 a",
                ".s-title-instructions-style h2 a span"
            ]
            
            title_text = None
            for sel in title_selectors:
                title_tag = item.select_one(sel)
                if title_tag:
                    title_text = title_tag.get_text(strip=True)
                    break
            
            if not title_text:
                img = item.select_one("img")
                title_text = img.get("alt", "").strip() if img else None
            
            product_name = clean_product_name(title_text)
            net_value, net_unit = split_quantity(title_text)
            
            # Get price
            price_selectors = [".a-price .a-offscreen", ".a-price-whole", ".a-price-range"]
            price = None
            for sel in price_selectors:
                price_elem = item.select_one(sel)
                if price_elem:
                    price = price_elem.get_text(strip=True)
                    break
            
            # Get rating
            rating_elem = item.select_one(".a-icon-alt")
            rating = rating_elem.get_text(strip=True) if rating_elem else None
            
            # Get product URL
            link_selectors = ["h2 a", "a.a-link-normal.s-no-outline", "a.a-link-normal"]
            product_url = None
            for sel in link_selectors:
                link = item.select_one(sel)
                if link and link.get("href"):
                    href = link["href"]
                    if not href.startswith("http"):
                        product_url = "https://www.amazon.in" + href
                    else:
                        product_url = href
                    break
            
            # Get image URL
            img = item.select_one("img")
            image_url = img.get("src") if img else None
            
            products.append({
                "product_id": asin,
                "product_name": product_name or f"Product {asin}",
                "category": "Food & Beverages",
                "mrp_value": price,
                "net_quantity_value": net_value,
                "net_quantity_unit": net_unit,
                "country_of_origin": None,
                "consumer_care_contact": None,
                "rating": rating,
                "url": product_url,
                "image_url": image_url
            })
            
        except Exception as e:
            print(f"Error parsing product item: {e}")
            continue
    
    return products

def clean_url_and_asin(href):
    """Extract clean URL and ASIN"""
    if not href:
        return None, None
    
    # Extract ASIN from various Amazon URL formats
    patterns = [
        r'/dp/([A-Z0-9]{10})',
        r'/product/([A-Z0-9]{10})',
        r'asin=([A-Z0-9]{10})',
        r'/([A-Z0-9]{10})(?:/|$|\?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, href)
        if match:
            asin = match.group(1)
            clean_url = f"https://www.amazon.in/dp/{asin}"
            return clean_url, asin
    
    return href, None

def normalize_to_hires(url):
    """Normalize image URL to high resolution"""
    if not isinstance(url, str):
        return url
    return re.sub(r'\.(SX|SY|SS|UX|UY|US|AC|QL|CR|SL)\d+([A-Z0-9_,\-]+)?', '._SL1500', url)

def extract_js_object_after_key(text, key):
    """Extract JavaScript object after a specific key"""
    pattern = rf'(["\']){re.escape(key)}\1\s*:\s*\{{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1
    depth = 0
    i = start
    
    while i < len(text):
        char = text[i]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
        i += 1
    
    return None

def extract_labeled_images(html):
    """Extract product images with labels from Amazon page"""
    soup = BeautifulSoup(html, "html.parser")
    images = []
    
    # Method 1: Look for colorImages in script tags (most accurate)
    for script in soup.find_all("script"):
        script_text = script.get_text(" ", strip=False)
        if "colorImages" not in script_text:
            continue
        
        try:
            # Extract the colorImages object
            obj = extract_js_object_after_key(script_text, "colorImages")
            if not obj:
                continue
            
            # Look for initial array
            match = re.search(r'(["\'])initial\1\s*:\s*\[', obj)
            if not match:
                continue
            
            # Extract the array content
            start = match.end() - 1
            depth = 0
            i = start
            while i < len(obj):
                if obj[i] == '[':
                    depth += 1
                elif obj[i] == ']':
                    depth -= 1
                    if depth == 0:
                        arr_content = obj[start:i+1]
                        break
                i += 1
            else:
                continue
            
            # Parse individual image objects
            parts = []
            current = []
            depth = 0
            
            for char in arr_content:
                current.append(char)
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        parts.append("".join(current))
                        current = []
            
            # Extract images with proper labels
            for part in parts:
                # Extract variant (label)
                variant_match = re.search(r'["\']variant["\']\s*:\s*["\']([^"\']+)', part)
                label = variant_match.group(1) if variant_match else "UNKNOWN"
                
                # Extract image URL (try hiRes first, then large, then main)
                url = None
                for key in ['hiRes', 'large']:
                    url_match = re.search(rf'["\']{key}["\']\s*:\s*["\']([^"\']+)', part)
                    if url_match:
                        url = url_match.group(1)
                        break
                
                # If no hiRes/large, try main object
                if not url:
                    main_match = re.search(r'["\']main["\']\s*:\s*\{([^}]+)', part)
                    if main_match:
                        main_content = main_match.group(1)
                        first_url = re.search(r'["\']([^"\']*https?[^"\']+)', main_content)
                        if first_url:
                            url = first_url.group(1)
                
                if url and url.startswith('http'):
                    images.append({
                        "label": label.upper(),
                        "url": normalize_to_hires(url)
                    })
                    
        except Exception as e:
            print(f"Error extracting colorImages: {e}")
            continue
    
    # Method 2: Alternative script parsing for image data
    if not images:
        for script in soup.find_all("script"):
            script_text = script.get_text(" ", strip=False)
            if "ImageBlockATF" in script_text or "imageBlock" in script_text:
                try:
                    # Find all Amazon image URLs in the script
                    img_urls = re.findall(r'["\']([^"\']*https://[^"\']*amazon[^"\']*\.(?:jpg|jpeg|png)[^"\']*)', script_text)
                    
                    # Deduplicate and filter
                    unique_urls = []
                    seen = set()
                    for url in img_urls:
                        normalized = normalize_to_hires(url)
                        if normalized not in seen and len(normalized) > 50:  # Filter out small/invalid URLs
                            seen.add(normalized)
                            unique_urls.append(normalized)
                    
                    # Label the images
                    for i, url in enumerate(unique_urls[:6]):  # Limit to 6 images
                        if i == 0:
                            label = "MAIN"
                        elif "back" in url.lower() or i == len(unique_urls) - 1:  # Last image often back
                            label = "BACK"
                        else:
                            label = f"PT{i:02d}"  # PT01, PT02, etc.
                        
                        images.append({
                            "label": label,
                            "url": url
                        })
                        
                    if images:
                        break
                        
                except Exception as e:
                    print(f"Error in alternative script parsing: {e}")
                    continue
    
    # Method 3: Fallback - look for images in img tags
    if not images:
        print("Using fallback image extraction from img tags")
        img_tags = soup.find_all("img", src=True)
        
        for img in img_tags:
            src = img.get("src", "")
            
            # Filter for Amazon product images
            if (src and "amazon" in src and 
                any(ext in src for ext in ['.jpg', '.jpeg', '.png']) and
                not any(skip in src.lower() for skip in ['sprite', 'icon', 'logo', 'button']) and
                len(src) > 50):  # Filter out small URLs
                
                # Determine label based on image context
                alt_text = img.get("alt", "").lower()
                parent_text = ""
                
                # Check parent elements for context
                parent = img.parent
                for _ in range(3):  # Check up to 3 levels up
                    if parent:
                        parent_text += parent.get_text(" ", strip=True).lower()
                        parent = parent.parent
                    else:
                        break
                
                # Determine label based on context
                if ("back" in alt_text or "back" in parent_text or 
                    "rear" in alt_text or "ingredients" in alt_text):
                    label = "BACK"
                elif len(images) == 0:
                    label = "MAIN"
                else:
                    label = f"PT{len(images)+1:02d}"
                
                images.append({
                    "label": label,
                    "url": normalize_to_hires(src)
                })
                
                if len(images) >= 6:  # Limit to 6 images
                    break
    
    print(f"Extracted {len(images)} images: {[img['label'] for img in images]}")
    return images

def preprocess_image(image):
    """Preprocess image for better OCR"""
    try:
        image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda p: p > 128 and 255)
        return image
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return image

def ocr_text_from_url(url, lang="eng+hin"):
    """Extract text from image URL using OCR"""
    if not isinstance(url, str) or not url.startswith("http"):
        return ""
    
    try:
        print(f"Performing OCR on: {url}")
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert("RGB")
        processed_img = preprocess_image(img)
        
        text = pytesseract.image_to_string(processed_img, lang=lang)
        print(f"OCR extracted {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"OCR error for {url}: {e}")
        return ""

def pick_main_and_back(product_url):
    """Get main and back product images with improved back image detection"""
    try:
        print(f"Extracting images from: {product_url}")
        html = get_html(product_url, render=True, premium=True)
        labeled_images = extract_labeled_images(html)
        
        if not labeled_images:
            print("No images found")
            return None, None
        
        print(f"Available images: {[(img['label'], img['url'][-50:]) for img in labeled_images]}")
        
        # Step 1: Find main image
        main_url = None
        for img in labeled_images:
            if img["label"].upper() in ["MAIN", "PT01"]:
                main_url = img["url"]
                break
        
        # If no main found, use first image
        if not main_url:
            main_url = labeled_images[0]["url"]
        
        # Step 2: Find back image with improved logic
        back_url = None
        
        # Priority 1: Look for explicitly labeled back images
        back_keywords = ["BACK", "REAR", "INGREDIENT", "NUTRITION", "INFO"]
        for img in labeled_images:
            label_upper = img["label"].upper()
            if any(keyword in label_upper for keyword in back_keywords):
                back_url = img["url"]
                print(f"Found back image by label: {label_upper}")
                break
        
        # Priority 2: If we have SWATCH images, use the last one (often back)
        if not back_url:
            swatch_images = [img for img in labeled_images if "SWATCH" in img["label"].upper()]
            if swatch_images:
                back_url = swatch_images[-1]["url"]
                print(f"Using last swatch as back image: {swatch_images[-1]['label']}")
        
        # Priority 3: Use OCR-based selection for remaining images
        if not back_url and len(labeled_images) > 1:
            print("Using OCR-based back image selection...")
            
            # Get candidate images (exclude main)
            candidates = [img for img in labeled_images if img["url"] != main_url][:5]  # Max 5 to check
            
            if candidates:
                print(f"Checking {len(candidates)} candidate images for text density...")
                
                # Calculate text density for each candidate
                scores = []
                for img in candidates:
                    try:
                        density = ocr_text_density(img["url"])
                        scores.append((density, img))
                        print(f"  {img['label']}: {density} words")
                    except Exception as e:
                        print(f"  {img['label']}: Error calculating density - {e}")
                        scores.append((0, img))
                
                # Sort by text density (highest first)
                scores.sort(reverse=True, key=lambda x: x[0])
                
                # Select image with highest text density (likely back with ingredients/info)
                if scores and scores[0][0] > 5:  # Minimum threshold of 5 words
                    back_url = scores[0][1]["url"]
                    print(f"Selected back image: {scores[0][1]['label']} (density: {scores[0][0]})")
                else:
                    # Fallback: use last image if no good text density found
                    back_url = candidates[-1]["url"]
                    print(f"Using last candidate as back image: {candidates[-1]['label']}")
        
        # Priority 4: Final fallback - use second image if available
        if not back_url and len(labeled_images) > 1:
            back_url = labeled_images[1]["url"]
            print(f"Using second image as back: {labeled_images[1]['label']}")
        
        print(f"Selected - Main: {bool(main_url)}, Back: {bool(back_url)}")
        return main_url, back_url
        
    except Exception as e:
        print(f"Error getting images for {product_url}: {e}")
        return None, None

def ocr_text_density(img_url):
    """Calculate text density in an image (words per image)"""
    try:
        print(f"    Calculating text density for: {img_url[-50:]}")
        response = requests.get(img_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Resize large images for faster processing
        max_width = 800
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Preprocess for better OCR
        processed_img = preprocess_image(img)
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, lang="eng+hin")
        
        # Count meaningful words (filter out single characters and numbers)
        words = [word.strip() for word in text.split() if len(word.strip()) > 2]
        word_count = len(words)
        
        print(f"    Text density: {word_count} words")
        return word_count
        
    except Exception as e:
        print(f"    Error calculating text density: {e}")
        return 0

def get_nlp_model(lang):
    """Load spaCy NLP model"""
    if lang in loaded_models:
        return loaded_models[lang]
    
    model_name = "en_core_web_sm" if lang == "en" else "en_core_web_sm"  # Use English as fallback
    
    try:
        nlp_model = spacy.load(model_name)
        loaded_models[lang] = nlp_model
        return nlp_model
    except OSError:
        print(f"SpaCy model '{model_name}' not found. Using basic extraction.")
        return None

def detect_language(text):
    """Detect text language"""
    if not text or not text.strip():
        return "en"
    
    try:
        # Check for Hindi characters
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        
        detected = detect(text)
        return detected if detected in ["en", "hi"] else "en"
    except LangDetectException:
        return "en"

def normalize_whitespace(text):
    """Normalize whitespace in text"""
    return re.sub(r"\s+", " ", text).strip()

# Country validation
try:
    VALID_COUNTRIES = {c.name.lower(): c.name for c in pycountry.countries}
    VALID_COUNTRIES.update({c.alpha_2.lower(): c.name for c in pycountry.countries})
    VALID_COUNTRIES.update({c.alpha_3.lower(): c.name for c in pycountry.countries})
except:
    VALID_COUNTRIES = {"india": "India", "in": "India", "ind": "India"}

def clean_country(candidate):
    """Clean and validate country name"""
    if not candidate:
        return "India"
    
    clean = candidate.strip().lower()
    if clean in VALID_COUNTRIES:
        return VALID_COUNTRIES[clean]
    
    # Look for partial matches
    for name in VALID_COUNTRIES:
        if name in clean or clean in name:
            return VALID_COUNTRIES[name]
    
    return "India"

# Regex patterns for contact extraction
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"[\+]?[\d\s\-\(\)]{10,}")

def extract_fields(front_text, back_text):
    """Extract compliance fields from OCR text"""
    combined_text = (back_text or "") + "\n\n" + (front_text or "")
    
    if not combined_text.strip():
        return {
            "manufacturer_name": None,
            "country_of_origin": "India",
            "consumer_care_contact": None,
        }
    
    text = normalize_whitespace(combined_text)
    lang = detect_language(text)
    
    result = {
        "manufacturer_name": None,
        "country_of_origin": None,
        "consumer_care_contact": None,
    }
    
    # Extract manufacturer using NLP
    nlp_model = get_nlp_model(lang)
    if nlp_model:
        try:
            doc = nlp_model(text[:1000])  # Limit text length for processing
            for ent in doc.ents:
                if ent.label_ == "ORG" and not result["manufacturer_name"]:
                    org_name = re.sub(r'[^a-zA-Z0-9\s.]', '', ent.text).strip()
                    if len(org_name.split()) > 1:
                        result["manufacturer_name"] = org_name.lower()
                        break
        except Exception as e:
            print(f"NLP processing error: {e}")
    
    # Fallback manufacturer extraction
    if not result["manufacturer_name"]:
        patterns = [
            r"(?:manufactured|packed|marketed|imported)\s*by\s*:?\s*(.+?)(?:[,\n]|$)",
            r"mfg\.?\s*by\s*:?\s*(.+?)(?:[,\n]|$)",
            r"mfd\.?\s*by\s*:?\s*(.+?)(?:[,\n]|$)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                manufacturer = match.group(1).strip()
                if len(manufacturer) > 3:  # Minimum length check
                    result["manufacturer_name"] = manufacturer.lower()
                    break
    
    # Extract country of origin
    country_patterns = [
        r"country\s*(?:of)?\s*origin\s*:?\s*([a-z\s.,-]+)",
        r"origin\s*:?\s*([a-z\s.,-]+)",
        r"made\s*in\s*([a-z\s.,-]+)"
    ]
    
    for pattern in country_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            result["country_of_origin"] = clean_country(match.group(1))
            break
    
    if not result["country_of_origin"]:
        result["country_of_origin"] = "India"
    
    # Extract contact information
    emails = re.findall(EMAIL_RE, text)
    phones = re.findall(PHONE_RE, text)
    
    contact_parts = []
    if emails:
        contact_parts.append("Email: " + emails[0])
    if phones:
        # Clean phone number
        clean_phone = re.sub(r'[^\d+]', '', phones[0])
        if len(clean_phone) >= 10:
            contact_parts.append("Phone: " + clean_phone)
    
    if contact_parts:
        result["consumer_care_contact"] = ", ".join(contact_parts[:2])  # Limit to 2 contacts
    
    return result

def is_missing(value):
    """Check if a value is missing or empty"""
    if value is None:
        return True
    
    str_val = str(value).strip().lower()
    return str_val in ("", "none", "nan", "nat", "null")

# ========================================
# FLASK API ENDPOINTS
# ========================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "Legal Metrology Compliance Checker API",
        "version": "1.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/check-compliance', methods=['POST'])
def check_compliance():
    """Main compliance checking endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        urls = data.get('urls', [])
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400
        
        print(f"Processing {len(urls)} URLs...")
        results = []
        
        for url_index, original_url in enumerate(urls):
            print(f"\n--- Processing URL {url_index + 1}/{len(urls)}: {original_url} ---")
            
            try:
                # Step 1: Get HTML
                page_html = get_html(original_url, render=True, premium=True)
                
                # Step 2: Try to parse as search page
                products_from_page = parse_search_page(page_html)
                
                # Step 3: If no products found, treat as individual product page
                if not products_from_page:
                    clean_url, asin = clean_url_and_asin(original_url)
                    if asin:
                        # Create basic product info for individual product page
                        soup = BeautifulSoup(page_html, "html.parser")
                        
                        # Try to extract product name from title or h1
                        title_selectors = ["#productTitle", "h1", "title"]
                        product_name = None
                        for sel in title_selectors:
                            elem = soup.select_one(sel)
                            if elem:
                                product_name = clean_product_name(elem.get_text(strip=True))
                                break
                        
                        products_from_page = [{
                            "product_id": asin,
                            "product_name": product_name or f"Product {asin}",
                            "category": "Food & Beverages",
                            "mrp_value": None,
                            "net_quantity_value": None,
                            "net_quantity_unit": None,
                            "country_of_origin": None,
                            "consumer_care_contact": None,
                            "rating": None,
                            "url": clean_url or original_url,
                            "image_url": None
                        }]
                
                if not products_from_page:
                    results.append({
                        "product_id": "ERROR",
                        "product_name": "Could not extract product information",
                        "mrp_value": None,
                        "net_quantity_value": None,
                        "net_quantity_unit": None,
                        "country_of_origin": None,
                        "manufacturer_name": None,
                        "consumer_care_contact": None,
                        "rating": None,
                        "url": original_url,
                        "main_image_url": None,
                        "back_image_url": None,
                        "compliance_score": "0%",
                        "violations": ["Product information extraction failed"]
                    })
                    continue
                
                print(f"Found {len(products_from_page)} products")
                
                # Step 4: Process each product
                for product_index, product in enumerate(products_from_page):
                    print(f"  Processing product {product_index + 1}: {product.get('product_name', 'Unknown')}")
                    
                    try:
                        # Get images
                        main_image_url, back_image_url = None, None
                        if product.get("url"):
                            main_image_url, back_image_url = pick_main_and_back(product["url"])
                        
                        # Perform OCR
                        front_text = ""
                        back_text = ""
                        
                        if main_image_url:
                            front_text = ocr_text_from_url(main_image_url, lang="eng+hin")
                        
                        if back_image_url:
                            back_text = ocr_text_from_url(back_image_url, lang="eng+hin")
                        
                        # Extract compliance fields
                        compliance_fields = extract_fields(front_text, back_text)
                        product.update(compliance_fields)
                        
                        # Calculate compliance score
                        required_fields = ["manufacturer_name", "country_of_origin", "consumer_care_contact"]
                        missing_count = sum(1 for field in required_fields if is_missing(product.get(field)))
                        compliance_percentage = max(0, 100 - (missing_count / len(required_fields) * 100))
                        compliance_score = f"{compliance_percentage:.0f}%"
                        
                        # Create violations list
                        violations = []
                        if is_missing(product.get("mrp_value")):
                            violations.append("Missing MRP information")
                        if is_missing(product.get("net_quantity_value")):
                            violations.append("Missing net quantity details")
                        if is_missing(product.get("manufacturer_name")):
                            violations.append("Missing manufacturer information")
                        if is_missing(product.get("consumer_care_contact")):
                            violations.append("Missing consumer care details")
                        
                        # Create final result
                        result = {
                            "product_id": product.get("product_id"),
                            "product_name": product.get("product_name"),
                            "mrp_value": product.get("mrp_value"),
                            "net_quantity_value": product.get("net_quantity_value"),
                            "net_quantity_unit": product.get("net_quantity_unit"),
                            "country_of_origin": product.get("country_of_origin"),
                            "manufacturer_name": product.get("manufacturer_name"),
                            "consumer_care_contact": product.get("consumer_care_contact"),
                            "rating": product.get("rating"),
                            "url": product.get("url", original_url),
                            "main_image_url": main_image_url,
                            "back_image_url": back_image_url,
                            "compliance_score": compliance_score,
                            "violations": violations if violations else ["None"]
                        }
                        
                        results.append(result)
                        print(f"    ✓ Completed: Compliance Score = {compliance_score}")
                        
                    except Exception as product_error:
                        print(f"    ✗ Error processing product: {product_error}")
                        results.append({
                            "product_id": product.get("product_id", "ERROR"),
                            "product_name": product.get("product_name", "Processing Error"),
                            "mrp_value": product.get("mrp_value"),
                            "net_quantity_value": product.get("net_quantity_value"),
                            "net_quantity_unit": product.get("net_quantity_unit"),
                            "country_of_origin": None,
                            "manufacturer_name": None,
                            "consumer_care_contact": None,
                            "rating": product.get("rating"),
                            "url": product.get("url", original_url),
                            "main_image_url": None,
                            "back_image_url": None,
                            "compliance_score": "0%",
                            "violations": [f"Processing error: {str(product_error)[:100]}"]
                        })
                        
            except Exception as url_error:
                print(f"✗ Error processing URL: {url_error}")
                traceback.print_exc()
                results.append({
                    "product_id": "ERROR",
                    "product_name": "URL Processing Error",
                    "mrp_value": None,
                    "net_quantity_value": None,
                    "net_quantity_unit": None,
                    "country_of_origin": None,
                    "manufacturer_name": None,
                    "consumer_care_contact": None,
                    "rating": None,
                    "url": original_url,
                    "main_image_url": None,
                    "back_image_url": None,
                    "compliance_score": "0%",
                    "violations": [f"URL processing error: {str(url_error)[:100]}"]
                })
        
        # Calculate summary statistics
        total_products = len(results)
        compliant_products = len([r for r in results if r.get("compliance_score") == "100%"])
        non_compliant_products = total_products - compliant_products
        
        print(f"\n=== SUMMARY ===")
        print(f"Total products processed: {total_products}")
        print(f"Fully compliant: {compliant_products}")
        print(f"Non-compliant: {non_compliant_products}")
        
        return jsonify({
            "success": True,
            "total_products": total_products,
            "compliant_products": compliant_products,
            "non_compliant_products": non_compliant_products,
            "results": results
        })
        
    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "total_products": 0,
            "compliant_products": 0,
            "non_compliant_products": 0,
            "results": []
        }), 500

@app.route('/download-csv', methods=['POST'])
def download_csv():
    """Generate and download CSV report"""
    try:
        data = request.json
        results = data.get('results', [])
        
        if not results:
            return jsonify({"error": "No results provided"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Ensure all required columns exist
        required_columns = [
            "product_id", "product_name", "mrp_value", "net_quantity_value",
            "net_quantity_unit", "country_of_origin", "manufacturer_name",
            "consumer_care_contact", "rating", "url", "main_image_url",
            "back_image_url", "compliance_score", "violations"
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Reorder columns
        df = df[required_columns]
        
        # Convert violations list to string
        if 'violations' in df.columns:
            df['violations'] = df['violations'].apply(lambda x: '; '.join(x) if isinstance(x, list) else str(x))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_filename = temp_file.name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f'legal_metrology_compliance_report_{timestamp}.csv'
        
        return send_file(
            temp_filename,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
        
    except Exception as e:
        print(f"CSV download error: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Legal Metrology Compliance Checker Backend...")
    print("\n📋 Required Installation Steps:")
    print("1. Install Python packages:")
    print("   pip install flask flask-cors requests pandas pytesseract pillow pycountry spacy langdetect beautifulsoup4")
    print("\n2. Install Tesseract OCR:")
    print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   - macOS: brew install tesseract")
    print("   - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    print("\n3. Download spaCy language models:")
    print("   python -m spacy download en_core_web_sm")
    print("   python -m spacy download hi_core_web_sm  # Optional for Hindi")
    print("\n4. Set up ScraperAPI key in the code (line 21)")
    print("\n🌐 Backend will run on http://localhost:5000")
    print("📝 API Endpoints:")
    print("   GET  /health - Health check")
    print("   POST /check-compliance - Main compliance checker")
    print("   POST /download-csv - Generate CSV report")
    print("\n🔧 Main fixes applied:")
    print("   ✓ Fixed indentation errors")
    print("   ✓ Added proper error handling and logging")
    print("   ✓ Improved Amazon parsing with fallback selectors")
    print("   ✓ Enhanced OCR text extraction")
    print("   ✓ Better image URL normalization")
    print("   ✓ Robust field extraction with multiple patterns")
    print("   ✓ Added fallback mechanisms for missing data")
    print("   ✓ Improved contact information extraction")
    print("   ✓ Better compliance score calculation")
    print("   ✓ Enhanced CSV generation")
    print("\n🎯 Ready to process Amazon product URLs!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
