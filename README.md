# Automated Compliance Checker for Legal Metrology Declarations  

## Overview  
This project implements a prototype Automated Compliance Checker for verifying compliance with the **Legal Metrology (Packaged Commodities) Rules, 2011** on Indian e-commerce platforms.  

The pipeline crawls product listings, extracts product details and packaging images, applies **OCR + NLP** techniques, and validates mandatory fields such as:  
- Manufacturer / Packer / Importer name  
- Net Quantity  
- MRP (inclusive of all taxes)  
- Country of Origin  
- Consumer Care Details (Email / Phone)  
- (Future: Full Address & Date of Manufacture / Import)  

The system outputs **compliance flags** and assigns a **compliance score** to each product based on completeness of declarations.  

---

## Workflow  

### 1. Data Acquisition  
- Scraped product listings using **ScraperAPI + BeautifulSoup**.  
- Extracted structured fields such as product ID, product name, net quantity, MRP, ratings, and product URLs.  
- Stored this structured data for later merging with image-based results.  

### 2. Image Collection  
- Collected both **main images** and **back/label images** from product pages.  
- Used a **text density heuristic** to select the most relevant label/back image:  
  - Download candidate images.  
  - Run OCR on each.  
  - Count the number of words extracted.  
  - Select the image with the **highest OCR word count** as the likely back-of-pack image.  

This allowed us to automatically choose packaging images that contained declarations such as manufacturer, MRP, and country of origin.  

### 3. OCR & NLP Processing  
- Applied **image preprocessing** to improve OCR accuracy:  
  - Converted images to grayscale.  
  - Increased contrast using `ImageEnhance`.  
  - Applied sharpening filters.  
  - Binarized the image (converted to pure black & white).  
- Ran **Tesseract OCR** with multilingual support (English + Hindi).  
- Extracted declarations from OCR text using a mix of **spaCy NLP** and **regex patterns**:  
  - Manufacturer name detected using `ORG` entities and custom rules.  
  - Country of origin matched against the ISO country list.  
  - Consumer care details extracted using regex for phone numbers and email addresses.  

### 4. Compliance Validation  
- Implemented a **rule-based engine** to check mandatory fields.  
- Rules included presence checks, format validation, and normalization (e.g., units for quantity, valid price formats).  
- Assigned each product a **compliance score** based on how many required fields were present.  
- Flagged violations where required fields were missing or invalid.  

---


## Planned Enhancements  

### Gemini API Integration  
Our current OCR + regex extraction works but faces challenges with:  
- Low-resolution or noisy images  
- Complex label layouts  
- Multilingual or mixed-language packaging  
- Extraction of detailed fields such as **full manufacturer address** and **date of manufacture/import**  

We plan to integrate **Gemini Vision + Language APIs** to:  
- Improve OCR accuracy on low-quality or multilingual labels.  
- Extract structured fields like manufacturer address and manufacturing/import dates more reliably.  
- Handle multilingual text (e.g., English + Hindi/Telugu on the same label).  
- Reduce reliance on heavy preprocessing steps and accelerate the pipeline.  

---

## Tech Stack  
- **Python**: requests, BeautifulSoup, pandas, regex  
- **OCR**: Tesseract OCR (multilingual)  
- **NLP**: spaCy (English and Hindi models)  
- **Data Processing**: pandas, pycountry, regex heuristics  
- **Planned AI**: Gemini Vision + Language API  

---

## Next Steps  
- Extend scraping to additional product categories (e.g., packaged snacks, beverages).  
- Integrate Gemini API for advanced OCR + NLP.  
- Extract and validate **full manufacturer address** and **date of manufacture/import**.  
- Build a dashboard (FastAPI + React) for real-time compliance visualization.  

---

## Impact  
- Enables scalable monitoring of Legal Metrology compliance.  
- Protects consumers from incomplete or misleading product information.  
- Provides regulators with evidence-backed violation reports.  
- Can integrate with e-commerce platforms for seller-side validation.  
