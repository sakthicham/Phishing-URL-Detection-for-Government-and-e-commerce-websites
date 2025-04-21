import pandas as pd
import numpy as np
import re
import urllib.parse
import joblib
import os
import tldextract
from flask import Flask, request, jsonify, render_template
from scipy import sparse
import datetime
import time
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phishing_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('phishing_detector')

app = Flask(__name__)

# Paths to dataset files
GOV_LEGIT_PATH = "Dataset/legit_gov_sites.csv"
GOV_PHISH_PATH = "Dataset/synthetic_phishing_gov_urls.csv"
ECOM_LEGIT_PATH = "Dataset/legit_ecommerce_urls.csv"
ECOM_PHISH_PATH = "Dataset/synthetic_phishing_ecommerce_urls.csv"
MODELS_DIR = "models"

# Global variables for model components
model = None
tfidf = None
last_dataset_modification_time = 0
retraining_lock = threading.Lock()
is_retraining = False

# Ensure datasets and model directories exist
def ensure_directories():
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(GOV_LEGIT_PATH), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Create empty datasets if they don't exist
    for path in [GOV_LEGIT_PATH, GOV_PHISH_PATH, ECOM_LEGIT_PATH, ECOM_PHISH_PATH]:
        if not os.path.exists(path):
            pd.DataFrame(columns=["url"]).to_csv(path, index=False)
            logger.info(f"Created empty dataset: {path}")

# URL features extraction (improved for better detection)
def extract_url_features(urls):
    features = np.zeros((len(urls), 22), dtype=np.float64)
    
    for i, url in enumerate(urls):
        try:
            # Skip invalid URLs
            if not url or not isinstance(url, str):
                continue
                
            # URL length (longer URLs are more suspicious)
            features[i, 0] = len(url)
            
            # Number of dots in the URL (more subdomains = more suspicious)
            features[i, 1] = url.count('.')
            
            # Number of special characters
            features[i, 2] = len(re.findall(r'[^a-zA-Z0-9.]', url))
            
            # Count of numbers in URL
            features[i, 3] = len(re.findall(r'\d', url))
            
            # Extract domain components
            try:
                extracted = tldextract.extract(url)
                domain = extracted.domain
                tld = extracted.suffix
                subdomain = extracted.subdomain
                
                # Subdomain count
                features[i, 4] = subdomain.count('.') + 1 if subdomain else 0
                
                # IMPORTANT CHANGE: Check for legitimate government TLDs
                is_legit_gov = is_legitimate_government_site(url)
                features[i, 5] = 1 if is_legit_gov else 0
                
                # IMPORTANT CHANGE: This feature now checks for .gov in non-TLD position
                # ONLY if it's not a legitimate government site
                features[i, 6] = 1 if (('.gov' in subdomain or '.gov' in domain) and 
                                      not is_legit_gov and tld != 'gov') else 0
                
                # Check for dashes in domain (phishing indicator)
                features[i, 7] = domain.count('-')
                
                # Domain length (longer domains more suspicious)
                features[i, 8] = len(domain)
                
                # Subdomain length (longer subdomains more suspicious)
                features[i, 9] = len(subdomain)
                
                # IMPORTANT CHANGE: Don't count multiple TLDs for legitimate government sites
                if is_legit_gov:
                    features[i, 10] = 0  # Not suspicious for legitimate gov sites
                else:
                    # Multiple TLDs in URL (e.g., .gov.com)
                    features[i, 10] = sum(1 for tld_pattern in ['.gov', '.mil', '.com', '.org', '.net'] 
                                         if tld_pattern in url)
                    features[i, 10] = max(0, features[i, 10] - 1)  # Subtract the actual TLD
                
                # IMPORTANT CHANGE: Don't penalize legitimate government sites
                if is_legit_gov:
                    features[i, 11] = 0  # Not suspicious
                else:
                    # Check for mismatched domain parts
                    features[i, 11] = 1 if ('gov' in domain or 'gov' in subdomain) and tld != 'gov' else 0
            except Exception as e:
                # Handle tldextract errors gracefully
                logger.warning(f"Error extracting domain components for {url}: {e}")
                features[i, 4:12] = 0
            
            # Presence of suspicious terms
            suspicious_terms = [
                'login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm',
                'validate', 'authentication', 'password'
            ]
            for j, term in enumerate(suspicious_terms):
                # IMPORTANT CHANGE: Don't count these terms as suspicious for legitimate gov sites
                if is_legitimate_government_site(url):
                    features[i, 12 + j] = 0  # Not suspicious for legitimate gov sites
                else:
                    features[i, 12 + j] = 1 if term in url.lower() else 0
        
        except Exception as e:
            # Catch any other errors during feature extraction
            logger.error(f"Error extracting features for URL {url}: {e}")
            # Set all features to 0 for this URL to avoid breaking the pipeline
            features[i, :] = 0
    
    return sparse.csr_matrix(features)

# Train a simple baseline model when no model exists
def train_baseline_model():
    logger.info("Training baseline model...")
    
    # Create simple datasets if they don't exist or are empty
    ensure_directories()
    
    # Sample data for baseline model
    sample_legitimate = [
        "https://www.google.com",
        "https://www.amazon.com",
        "https://www.usa.gov",
        "https://www.irs.gov",
        "https://www.whitehouse.gov"
    ]
    
    sample_phishing = [
        "http://gov.income-tax-efiling.gov.verify-id.com",
        "https://www.irs-tax-refund.gov.com",
        "https://account-verify.gov.irs.login.com",
        "https://secure-login.amazonn.com",
        "https://verify-account.paypal.phishing.com"
    ]
    
    # Create simple dataframe
    df = pd.DataFrame({
        "url": sample_legitimate + sample_phishing,
        "label": [0] * len(sample_legitimate) + [1] * len(sample_phishing)
    })
    
    # Extract features
    X = df['url']
    y = df['label']
    
    # TF-IDF features
    new_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
    X_tfidf = new_tfidf.fit_transform(X)
    
    # URL features
    X_url_features = extract_url_features(X)
    
    # Combine features
    X_combined = sparse.hstack([X_tfidf, X_url_features])
    
    new_model = XGBClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=50,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    new_model.fit(X_combined, y)
    
    # Save the baseline model
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    model_name = f"phishing_detector_baseline_v{timestamp}"
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, f"{model_name}_tfidf.pkl")
    
    joblib.dump(new_model, model_path)
    joblib.dump(new_tfidf, tfidf_path)
    
    logger.info(f"Baseline model saved as {model_name}_model.pkl")
    
    return new_model, new_tfidf

# Load the latest model with better error handling
def load_latest_model():
    global model, tfidf
    
    try:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR, exist_ok=True)
            logger.info("Created models directory")
            return train_baseline_model()
        
        # Get all model files
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_model.pkl")]
        
        if not model_files:
            logger.info("No model files found. Training baseline model.")
            return train_baseline_model()
        
        # Sort by timestamp
        latest_model_file = sorted(model_files)[-1]
        model_base_name = latest_model_file.replace("_model.pkl", "")
        
        model_path = os.path.join(MODELS_DIR, latest_model_file)
        tfidf_path = os.path.join(MODELS_DIR, model_base_name + "_tfidf.pkl")
        
        # Check if both files exist
        if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
            logger.warning(f"Model or TF-IDF file missing. Training baseline model.")
            return train_baseline_model()
        
        # Load model components
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        
        logger.info(f"Loaded model: {latest_model_file}")
        return model, tfidf
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return train_baseline_model()

# Improved function to determine if a URL is government-related or e-commerce
def classify_url_type(url):
    try:
        # Handle invalid URLs
        if not url or not isinstance(url, str):
            return "ecommerce"  # Default to ecommerce for invalid URLs
            
        # Extract domain components
        extracted = tldextract.extract(url)
        domain = extracted.domain
        tld = extracted.suffix
        subdomain = extracted.subdomain
        
        # Government domain indicators (expanded)
        gov_indicators = [
            # Official TLDs
            '.gov', '.gov.', '.mil', '.fed.', '.gc.ca',
            
            # Government terms in domain and path
            'government', 'federal', 'agency', 'department', 
            'admin', 'official', 'whitehouse', 'senate', 'congress',
            
            # Tax and finance related government terms
            'irs', 'treasury', 'tax', 'revenue', 'income-tax',
            'efiling', 'e-filing', 'refund', 'stimulus',
            
            # Identity verification terms with gov context
            'verify-id', 'identity', 'passport', 'license'
        ]
        
        # Check if URL has a .gov or .mil TLD (strongest indicator)
        if tld == 'gov' or tld == 'mil':
            return "government"
        
        # Check for .gov appearing in subdomain or domain (phishing technique)
        if '.gov' in subdomain or '.gov' in domain or 'gov.' in subdomain or 'gov.' in domain:
            return "government"
        
        # Check for government terms in any part of the URL
        for term in gov_indicators:
            if term in url.lower():
                return "government"
        
        # E-commerce indicators
        ecom_indicators = [
            'shop', 'store', 'buy', 'sale', 'deal', 'price', 'checkout',
            'cart', 'order', 'payment', 'amazon', 'ebay', 'walmart', 
            'etsy', 'shopify', 'alibaba', 'product', 'purchase', 'ecommerce'
        ]
        
        # Check if URL is e-commerce related
        for term in ecom_indicators:
            if term in domain.lower() or term in url.lower():
                return "ecommerce"
        
        # Default to e-commerce if unsure
        return "ecommerce"
    
    except Exception as e:
        logger.error(f"Error classifying URL type: {e}")
        # Default to ecommerce on error
        return "ecommerce"

# Function to append URL to appropriate phishing dataset
def append_to_phishing_dataset(url, url_type):
    try:
        if url_type == "government":
            dataset_path = GOV_PHISH_PATH
        else:  # e-commerce
            dataset_path = ECOM_PHISH_PATH
        
        # Check if file exists, if not create it
        if not os.path.exists(dataset_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            
            # Create new CSV with header
            pd.DataFrame(columns=["url"]).to_csv(dataset_path, index=False)
        
        # Read the existing dataset
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            df = pd.DataFrame(columns=["url"])
        
        # Check if URL already exists
        if url in df["url"].values:
            logger.info(f"URL already exists in {dataset_path}")
            return False
        
        # Append the URL
        new_row = pd.DataFrame({"url": [url]})
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save the updated dataset
        try:
            df.to_csv(dataset_path, index=False)
            logger.info(f"URL appended to {dataset_path}")
            
            # Trigger retraining check
            check_for_retraining()
            
            return True
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error appending to phishing dataset: {e}")
        return False

# Check if datasets have been modified and retraining is needed
def check_for_retraining():
    global last_dataset_modification_time, is_retraining
    
    try:
        # Get the latest modification time of any dataset
        dataset_paths = [GOV_LEGIT_PATH, GOV_PHISH_PATH, ECOM_LEGIT_PATH, ECOM_PHISH_PATH]
        latest_mod_time = 0
        
        for path in dataset_paths:
            if os.path.exists(path):
                mod_time = os.path.getmtime(path)
                latest_mod_time = max(latest_mod_time, mod_time)
        
        # If we have new modifications and not already retraining
        if latest_mod_time > last_dataset_modification_time and not is_retraining:
            with retraining_lock:
                if not is_retraining:  # Double-check in case another thread started retraining
                    is_retraining = True
                    last_dataset_modification_time = latest_mod_time
                    
                    # Start retraining in a separate thread
                    threading.Thread(target=retrain_model).start()
    except Exception as e:
        logger.error(f"Error checking for retraining: {e}")

# Function to retrain the model
def retrain_model():
    global model, tfidf, is_retraining
    
    try:
        logger.info("Starting model retraining...")
        
        # -------- Load all datasets --------
        logger.info("Loading datasets...")
        
        # Create datasets if they don't exist
        ensure_directories()
        
        # Load datasets with better error handling
        try:
            gov_legit = pd.read_csv(GOV_LEGIT_PATH)
            gov_phish = pd.read_csv(GOV_PHISH_PATH)
            ecom_legit = pd.read_csv(ECOM_LEGIT_PATH)
            ecom_phish = pd.read_csv(ECOM_PHISH_PATH)
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            # Use empty dataframes if loading fails
            gov_legit = pd.DataFrame(columns=["url"])
            gov_phish = pd.DataFrame(columns=["url"])
            ecom_legit = pd.DataFrame(columns=["url"])
            ecom_phish = pd.DataFrame(columns=["url"])

        # Add labels and source info
        gov_legit["label"] = 0
        gov_legit["source"] = "government"
        gov_phish["label"] = 1
        gov_phish["source"] = "government"
        ecom_legit["label"] = 0
        ecom_legit["source"] = "ecommerce"
        ecom_phish["label"] = 1
        ecom_phish["source"] = "ecommerce"

        # Combine all datasets
        df = pd.concat([gov_legit, gov_phish, ecom_legit, ecom_phish])
        df = df.drop_duplicates(subset=['url']).reset_index(drop=True)

        # Check if we have enough data
        if len(df) < 10 or len(df[df['label'] == 0]) < 2 or len(df[df['label'] == 1]) < 2:
            logger.warning("Not enough data for meaningful training. Using baseline model.")
            model, tfidf = train_baseline_model()
            with retraining_lock:
                is_retraining = False
            return

        # Print basic dataset info
        logger.info(f"Total dataset size: {len(df)}")
        logger.info(f"Legitimate URLs: {len(df[df['label'] == 0])}")
        logger.info(f"Phishing URLs: {len(df[df['label'] == 1])}")

        # -------- Train-Test Split --------
        X = df['url']
        y = df['label']
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            logger.error(f"Error in train-test split: {e}")
            # Fall back to non-stratified split if stratified fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Testing set size: {len(X_test)}")

        # -------- Feature Extraction --------
        # TF-IDF features
        logger.info("Extracting TF-IDF features...")
        new_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
        X_train_tfidf = new_tfidf.fit_transform(X_train)
        X_test_tfidf = new_tfidf.transform(X_test)

        # URL features
        logger.info("Extracting URL features...")
        X_train_url_features = extract_url_features(X_train)
        X_test_url_features = extract_url_features(X_test)

        # Combine features
        X_train_combined = sparse.hstack([X_train_tfidf, X_train_url_features])
        X_test_combined = sparse.hstack([X_test_tfidf, X_test_url_features])

        # -------- Train Model --------
        logger.info("Training XGBoost model...")
        new_model = XGBClassifier(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        new_model.fit(X_train_combined, y_train)

        # -------- Model Evaluation --------
        logger.info("Evaluating model on test set...")
        y_pred = new_model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Log evaluation metrics
        logger.info(f"True Positives: {tp}, False Positives: {fp}")
        logger.info(f"True Negatives: {tn}, False Negatives: {fn}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        # -------- Save Model with Versioning --------
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        model_name = f"phishing_detector_v{timestamp}"
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save model artifacts
        model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
        tfidf_path = os.path.join(MODELS_DIR, f"{model_name}_tfidf.pkl")
        metadata_path = os.path.join(MODELS_DIR, f"{model_name}_metadata.txt")
        
        joblib.dump(new_model, model_path)
        joblib.dump(new_tfidf, tfidf_path)
        
        # Save model metadata
        with open(metadata_path, "w") as f:
            f.write(f"Model trained on: {now}\n")
            f.write(f"Dataset size: {len(df)}\n")
            f.write(f"Training set size: {len(X_train)}\n")
            f.write(f"Test set size: {len(X_test)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"True Positives: {tp}, False Positives: {fp}\n")
            f.write(f"True Negatives: {tn}, False Negatives: {fn}\n")
            f.write(f"Model parameters: {new_model.get_params()}\n")

        # Update global model objects
        with retraining_lock:
            model = new_model
            tfidf = new_tfidf
        
        logger.info(f"Model saved as {model_name}_model.pkl")
        
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        # Fall back to baseline model if retraining fails
        try:
            model, tfidf = train_baseline_model()
        except Exception as inner_e:
            logger.error(f"Could not create baseline model: {inner_e}")
    finally:
        with retraining_lock:
            is_retraining = False

# Improved URL checking function with better error handling
def is_legitimate_government_site(url):
    """
    Enhanced function to determine if a URL is a legitimate government site.
    Returns True for legitimate government sites, False otherwise.
    """
    try:
        # Parse the URL
        extracted = tldextract.extract(url)
        domain = extracted.domain
        tld = extracted.suffix
        subdomain = extracted.subdomain
        full_domain = '.'.join(part for part in [subdomain, domain, tld] if part)
        
        # Whitelist of known legitimate government domains
        # Add more based on your specific needs and region
        legitimate_gov_domains = [
            # US Government
            '.gov.', '.mil.', 
            'usa.gov', 'whitehouse.gov', 'irs.gov', 'treasury.gov',
            'ssa.gov', 'medicare.gov', 'cms.gov', 'dhs.gov', 'fbi.gov',
            'state.gov', 'ed.gov', 'epa.gov', 'fda.gov', 'ftc.gov',
            'gsa.gov', 'nasa.gov', 'nih.gov', 'nist.gov', 'noaa.gov',
            'nsf.gov', 'usda.gov', 'va.gov',
            
            # Indian Government (add these for your demo)
            'india.gov.in', 'gov.in', 'nic.in', 'mygov.in', 'meity.gov.in',
            'incometaxindiaefiling.gov.in', 'epfindia.gov.in', 'uidai.gov.in',
            'digitalindia.gov.in', 'mohfw.gov.in', 'mha.gov.in', 'mea.gov.in',
            'cbic.gov.in', 'mof.gov.in', 'pib.gov.in', 'ibef.org', 'niti.gov.in',
            
            # Other countries (add relevant domains for your needs)
            'gc.ca', 'canada.ca', 'gov.uk', 'gov.au', 'gob.mx',
            'gouv.fr', 'bund.de', 'gobierno.es'
        ]
        
        # Check for direct match in whitelist
        for whitelist_domain in legitimate_gov_domains:
            if whitelist_domain in full_domain:
                return True
        
        # Check for .gov and .mil TLDs (strongest indicator of legitimate gov site)
        if tld in ['gov', 'mil']:
            # Additional check for country-specific government domains
            # For example, for Indian government sites
            if domain.endswith('.in') or tld.endswith('.in'):
                return True
            return True
            
        # Check for country-specific government domains (e.g., .gov.in for India)
        if tld.startswith('gov.') or (tld == 'in' and domain == 'gov'):
            return True
            
        # NIC (National Informatics Centre) domains in India are legitimate
        if tld == 'in' and domain == 'nic':
            return True
            
        # Check for specific international government patterns
        if (tld in ['ca', 'uk', 'au'] and (domain == 'gov' or subdomain == 'gov')):
            return True
            
        return False
        
    except Exception as e:
        logger.warning(f"Error in legitimate government check: {e}")
        return False


def check_url(url):
    """
    Enhanced function to check if a URL is likely to be phishing or legitimate
    with improved handling for government sites
    """
    global model, tfidf
    
    try:
        # Validate URL
        if not url or not isinstance(url, str):
            return {"error": "Invalid URL provided"}
            
        # Make sure we have a model loaded
        if model is None or tfidf is None:
            try:
                model, tfidf = load_latest_model()
            except Exception as e:
                logger.error(f"Could not load model: {str(e)}")
                return {"error": f"Could not load model: {str(e)}"}
        
        # Check whitelist for legitimate government sites first
        is_legit_gov = is_legitimate_government_site(url)
        
        # If it's a known legitimate government site, return immediately
        if is_legit_gov:
            return {
                "url": url,
                "is_phishing": False,
                "phishing_probability": 0.05,  # Very low probability
                "risk_level": "Low",
                "url_type": "government",
                "detection_details": {
                    "is_known_legitimate_gov": True,
                    "is_real_gov_tld": True,
                    "has_gov_in_domain": True, 
                    "multiple_tlds_detected": False,
                    "has_suspicious_gov_terms": False
                }
            }
        
        # If not in whitelist, continue with normal detection
        url_type_initial = classify_url_type(url)
        
        # Extract domain components for special rule checks
        try:
            extracted = tldextract.extract(url)
            domain = extracted.domain
            tld = extracted.suffix
            subdomain = extracted.subdomain
            
            # Special case flags
            is_real_gov = tld == 'gov' or tld == 'mil' or tld.startswith('gov.')
            
            # IMPORTANT CHANGE: Don't mark real .gov domains as suspicious
            has_fake_gov = ('gov' in domain or 'gov' in subdomain or '.gov' in url) and not is_real_gov
            
            # Multiple suspicious TLDs check
            tld_patterns = ['.gov', '.mil', '.com', '.org', '.net']
            multiple_tlds = sum(1 for pat in tld_patterns if pat in url) > 1
            
            # Government terms check - only consider suspicious in non-government domains
            gov_terms = ['tax', 'irs', 'treasury', 'income', 'refund', 'verify-id']
            has_gov_terms = any(term in url.lower() for term in gov_terms) and not is_real_gov
        except Exception as e:
            logger.warning(f"Error extracting URL components: {e}")
            # Set default values if domain extraction fails
            is_real_gov = False
            has_fake_gov = False
            multiple_tlds = False
            has_gov_terms = False
        
        # Extract features
        try:
            url_tfidf = tfidf.transform([url])
            url_features = extract_url_features([url])
            url_combined = sparse.hstack([url_tfidf, url_features])
            
            # Make prediction
            prediction = model.predict(url_combined)[0]
            probability = model.predict_proba(url_combined)[0][1]
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Fall back to rule-based detection if model prediction fails
            prediction = 1 if (has_fake_gov or multiple_tlds) else 0
            probability = 0.8 if (has_fake_gov or multiple_tlds) else 0.2
        
        # IMPORTANT CHANGE: Add stronger logic to prevent false positives on government sites
        # If URL has real .gov or .mil TLD, lower the probability regardless of model output
        if is_real_gov:
            # Override model for real government sites
            probability = min(probability, 0.2)  # Cap at 20% max
            prediction = 0  # Force classification as legitimate
        
        # Apply special rules for suspicious patterns only if it's not a real gov site
        if url_type_initial == "government" and not is_real_gov:
            # Apply stricter checks for fake government sites
            if has_fake_gov or multiple_tlds or '-' in domain:
                probability = max(probability, 0.85)  # Minimum 85% if suspicious gov domain
                prediction = 1  # Force classification as phishing
        
        # Special handling for URLs that try to blend government with verification/secure terms
        # But only if it's not a real government site
        if has_gov_terms and any(term in url.lower() for term in ['verify', 'secure', 'login', 'account']):
            if not is_real_gov:  # Not a legitimate .gov TLD
                probability = max(probability, 0.80)
                prediction = 1
        
        # Check for the specific pattern in your example
        if ('verify-id' in url or 'income-tax' in url or 'efiling' in url) and '.gov' in url and not is_real_gov:
            probability = 0.95  # Very likely phishing
            prediction = 1
            url_type_initial = "government"  # Force government classification
        
        result = {
            "url": url,
            "is_phishing": bool(prediction),
            "phishing_probability": float(probability),
            "risk_level": "High" if probability > 0.8 else "Medium" if probability > 0.5 else "Low",
            "url_type": url_type_initial
        }
        
        # Include more detailed information for debugging
        result["detection_details"] = {
            "is_known_legitimate_gov": is_legit_gov,
            "is_real_gov_tld": is_real_gov,
            "has_gov_in_domain": has_fake_gov,
            "multiple_tlds_detected": multiple_tlds,
            "has_suspicious_gov_terms": has_gov_terms
        }
        
        # If phishing with high probability, append to dataset
        if result["is_phishing"] and probability > 0.7:  # Only add if confidence is high
            append_success = append_to_phishing_dataset(url, url_type_initial)
            result["added_to_dataset"] = append_success
        else:
            result["added_to_dataset"] = False
        
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in check_url: {e}")
        return {"error": f"Error checking URL: {str(e)}"}


@app.route('/api/check', methods=['POST'])
def api_check():
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400
        
        url = data['url']
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        result = check_url(url)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in api_check endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    global is_retraining
    
    try:
        if is_retraining:
            return jsonify({"status": "already_running", "message": "Retraining is already in progress"}), 409
        
        with retraining_lock:
            if not is_retraining:
                is_retraining = True
                threading.Thread(target=retrain_model).start()
        
        return jsonify({"status": "started", "message": "Model retraining started"})
    except Exception as e:
        logger.error(f"Error in retrain endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    global model, tfidf, is_retraining
    
    try:
        # Get model info
        model_info = {}
        if model is not None:
            try:
                # Get the most recent model file
                model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_model.pkl")]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    model_base_name = latest_model.replace("_model.pkl", "")
                    metadata_path = os.path.join(MODELS_DIR, f"{model_base_name}_metadata.txt")
                    
                    # Get metadata if it exists
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = f.read()
                        model_info["metadata"] = metadata
                        model_info["name"] = latest_model
            except Exception as e:
                model_info["error"] = str(e)
        
        # Get dataset info
        dataset_info = {}
        for dataset_name, path in [
            ("government_legitimate", GOV_LEGIT_PATH),
            ("government_phishing", GOV_PHISH_PATH),
            ("ecommerce_legitimate", ECOM_LEGIT_PATH),
            ("ecommerce_phishing", ECOM_PHISH_PATH)
        ]:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    dataset_info[dataset_name] = {
                        "count": len(df),
                        "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    dataset_info[dataset_name] = {"count": 0, "last_modified": "N/A"}
            except Exception as e:
                dataset_info[dataset_name] = {"error": str(e)}
        
        return jsonify({
            "is_model_loaded": model is not None and tfidf is not None,
            "is_retraining": is_retraining,
            "model_info": model_info,
            "dataset_info": dataset_info
        })
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/datasets', methods=['GET'])
def api_datasets():
    try:
        dataset_info = {}
        
        for dataset_name, path in [
            ("government_legitimate", GOV_LEGIT_PATH),
            ("government_phishing", GOV_PHISH_PATH),
            ("ecommerce_legitimate", ECOM_LEGIT_PATH),
            ("ecommerce_phishing", ECOM_PHISH_PATH)
        ]:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    dataset_info[dataset_name] = {
                        "count": len(df),
                        "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S"),
                        "sample": df["url"].tolist()[:5] if len(df) > 0 else []
                    }
                else:
                    dataset_info[dataset_name] = {"count": 0, "last_modified": "N/A", "sample": []}
            except Exception as e:
                dataset_info[dataset_name] = {"error": str(e)}
        
        return jsonify(dataset_info)
    except Exception as e:
        logger.error(f"Error in datasets endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/add_phishing_url', methods=['POST'])
def api_add_phishing_url():
    """
    Manually add a URL to the phishing dataset
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400
        
        url = data['url']
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Determine URL type if not provided
        url_type = data.get('url_type')
        if not url_type:
            url_type = classify_url_type(url)
        
        # Append to dataset
        append_success = append_to_phishing_dataset(url, url_type)
        
        if append_success:
            return jsonify({"status": "success", "message": f"URL added to {url_type} phishing dataset"})
        else:
            return jsonify({"status": "error", "message": "Failed to add URL to dataset or URL already exists"}), 400
    except Exception as e:
        logger.error(f"Error in add_phishing_url endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/bulk_add_phishing_urls', methods=['POST'])
def api_bulk_add_phishing_urls():
    """
    Add multiple URLs to the phishing dataset at once
    """
    try:
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({"error": "No URLs provided"}), 400
        
        urls = data['urls']
        results = []
        
        for url in urls:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            # Determine URL type
            url_type = classify_url_type(url)
            
            # Append to dataset
            append_success = append_to_phishing_dataset(url, url_type)
            
            results.append({
                "url": url,
                "url_type": url_type,
                "added": append_success
            })
        
        # Trigger retraining only once after all URLs are added
        check_for_retraining()
        
        return jsonify({
            "status": "success",
            "total_urls": len(urls),
            "added_urls": sum(1 for r in results if r["added"]),
            "results": results
        })
    except Exception as e:
        logger.error(f"Error in bulk_add_phishing_urls endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/test_detection', methods=['GET'])
def api_test_detection():
    try:
        test_urls = [
            "http://gov.income-tax-efiling.gov.verify-id.com",
            "https://www.irs-tax-refund.gov.com",
            "https://account-verify.gov.irs.login.com",
            "https://www.amazon.com"    
        ]
        
        results = []
        for url in test_urls:
            try:
                result = check_url(url)
                results.append(result)
            except Exception as e:
                results.append({"url": url, "error": str(e)})
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in test_detection endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().isoformat()})

# Initialize on startup
with app.app_context():
    def initialize_app():
        """Initialize the application before handling the first request"""
        global model, tfidf
        
        # Ensure directories exist
        ensure_directories()
        
        # Try to load the model
        try:
            model, tfidf = load_latest_model()
            logger.info("Model loaded successfully during initialization")
        except Exception as e:
            logger.warning(f"Could not load model during initialization: {e}")
            logger.info("Will attempt to create baseline model")
            try:
                model, tfidf = train_baseline_model()
                logger.info("Baseline model created successfully")
            except Exception as inner_e:
                logger.error(f"Failed to create baseline model: {inner_e}")
                logger.warning("API will attempt to initialize model on first request")

# For command line usage
def cli():
    while True:
        print("\n===== Phishing URL Detector CLI =====")
        print("1. Check URL")
        print("2. Retrain model")
        print("3. Add URL to phishing dataset")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            url = input("Enter URL to check: ")
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            try:
                result = check_url(url)
                if "error" in result:
                    print(f"Error: {result['error']}")
                    continue
                    
                print("\nResult:")
                print(f"URL: {result['url']}")
                print(f"URL Type: {result['url_type']}")
                print(f"Is Phishing: {'Yes' if result['is_phishing'] else 'No'}")
                print(f"Probability: {result['phishing_probability']:.4f}")
                print(f"Risk Level: {result['risk_level']}")
                
                # Print detailed detection info
                if 'detection_details' in result:
                    print("\nDetection Details:")
                    for key, value in result['detection_details'].items():
                        print(f"  {key}: {value}")
                    
                if result.get('added_to_dataset'):
                    print("URL has been added to the phishing dataset.")
            except Exception as e:
                print(f"Error checking URL: {e}")
        
        elif choice == "2":
            print("Starting model retraining...")
            try:
                retrain_model()
                print("Retraining completed.")
            except Exception as e:
                print(f"Error during retraining: {e}")
        
        elif choice == "3":
            url = input("Enter phishing URL to add: ")
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
                
            try:
                url_type = classify_url_type(url)
                append_success = append_to_phishing_dataset(url, url_type)
                if append_success:
                    print(f"URL added to {url_type} phishing dataset.")
                else:
                    print("Failed to add URL or URL already exists in dataset.")
            except Exception as e:
                print(f"Error adding URL: {e}")
        
        elif choice == "4":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    # Make sure directories exist before starting the app
    ensure_directories()
    
    # Try to load or create a model before starting the server
    try:
        model, tfidf = load_latest_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        try:
            logger.info("Creating baseline model...")
            model, tfidf = train_baseline_model()
            logger.info("Baseline model created successfully")
        except Exception as inner_e:
            logger.error(f"Failed to create baseline model: {inner_e}")
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)