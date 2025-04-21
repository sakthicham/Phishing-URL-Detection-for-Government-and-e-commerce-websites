import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
import re
import urllib.parse
import joblib
import datetime
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# Force flush after every print to ensure output is visible
def custom_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# Create a directory for model versions
os.makedirs("models", exist_ok=True)

# -------- Load all datasets --------
custom_print("Loading datasets...")
try:
    gov_legit = pd.read_csv("Dataset/legit_gov_sites.csv")
    gov_phish = pd.read_csv("Dataset/synthetic_phishing_gov_urls.csv")
    ecom_legit = pd.read_csv("Dataset/legit_ecommerce_urls.csv")
    ecom_phish = pd.read_csv("Dataset/synthetic_phishing_ecommerce_urls.csv")
    custom_print("All datasets loaded successfully.")
except Exception as e:
    custom_print(f"Error loading datasets: {str(e)}")
    sys.exit(1)

# Add labels and source info
gov_legit["label"] = 0  # Legitimate
gov_legit["source"] = "government"
gov_legit["category"] = 0  # Government category
gov_phish["label"] = 1  # Phishing
gov_phish["source"] = "government"
gov_phish["category"] = 0  # Government category
ecom_legit["label"] = 0  # Legitimate
ecom_legit["source"] = "ecommerce"
ecom_legit["category"] = 1  # Ecommerce category
ecom_phish["label"] = 1  # Phishing
ecom_phish["source"] = "ecommerce"
ecom_phish["category"] = 1  # Ecommerce category

# Combine all datasets
df = pd.concat([gov_legit, gov_phish, ecom_legit, ecom_phish])
df = df.drop_duplicates(subset=['url']).reset_index(drop=True)

# Print basic dataset info
custom_print(f"Total dataset size: {len(df)}")
custom_print(f"Legitimate URLs: {len(df[df['label'] == 0])}")
custom_print(f"Phishing URLs: {len(df[df['label'] == 1])}")
custom_print(f"Government URLs: {len(df[df['category'] == 0])}")
custom_print(f"E-commerce URLs: {len(df[df['category'] == 1])}")

# -------- Create multi-class label (for combined prediction) --------
# 0 = Legitimate Government, 1 = Phishing Government
# 2 = Legitimate E-commerce, 3 = Phishing E-commerce
df['combined_label'] = df['category'] * 2 + df['label']

# -------- Train-Test Split for Separate Models --------
# For phishing/legitimate classification
X = df['url']
y_phish = df['label']
X_train_phish, X_test_phish, y_train_phish, y_test_phish = train_test_split(
    X, y_phish, test_size=0.2, random_state=42, stratify=y_phish
)

# For category classification
y_cat = df['category']
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X, y_cat, test_size=0.2, random_state=43, stratify=y_cat
)

# For combined classification
y_combined = df['combined_label']
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    X, y_combined, test_size=0.2, random_state=44, stratify=y_combined
)

custom_print(f"Training set size: {len(X_train_phish)}")
custom_print(f"Testing set size: {len(X_test_phish)}")

# -------- Feature Extraction --------
# TF-IDF features
custom_print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=15000)
X_train_tfidf_phish = tfidf.fit_transform(X_train_phish)
X_test_tfidf_phish = tfidf.transform(X_test_phish)

# Use same TF-IDF for the category model
X_train_tfidf_cat = tfidf.transform(X_train_cat)
X_test_tfidf_cat = tfidf.transform(X_test_cat)

# Use same TF-IDF for the combined model
X_train_tfidf_combined = tfidf.transform(X_train_combined)
X_test_tfidf_combined = tfidf.transform(X_test_combined)

def extract_url_features(urls):
    features = np.zeros((len(urls), 44), dtype=np.float64)  # Increased from 40 to 44
    
    # Commonly abused TLDs
    suspicious_tlds = ['.xyz', '.top', '.club', '.online', '.site', '.info']
    
    # Enhanced suspicious terms - expanded to catch more phishing patterns
    suspicious_terms = [
        'login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm',
        'support', 'billing', 'password', 'pay', 'payment', 'auth', 'authenticate',
        'security', 'wallet', 'official', 'customer'
    ]
    
    # Brand names commonly targeted by phishing
    ecommerce_brands = [
        'paypal', 'amazon', 'ebay', 'walmart', 'alibaba', 'shopify', 
        'netflix', 'etsy', 'bestbuy', 'target'
    ]
    
    government_terms = [
        'gov', 'government', 'federal', 'state', 'tax', 'irs', 'treasury',
        'usps', 'postal', 'medicare', 'socialsecurity', 'benefit', 'grant'
    ]
    
    for i, url in enumerate(urls):
        url_lower = url.lower()
        
        # URL length (phishing URLs tend to be longer)
        features[i, 0] = len(url)
        
        # Number of dots in the URL
        features[i, 1] = url.count('.')
        
        # Number of special characters
        features[i, 2] = len(re.findall(r'[^a-zA-Z0-9.]', url))
        
        # Count of numbers in URL
        features[i, 3] = len(re.findall(r'\d', url))
        
        # Parse URL parts
        try:
            parsed = urllib.parse.urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
            domain = parsed.netloc if parsed.netloc else url.split('/')[0]
            path = parsed.path
        except:
            domain = url.split('/')[0] if '/' in url else url
            path = url[len(domain):] if '/' in url else ''
        
        # Subdomain count (dots in domain)
        features[i, 4] = domain.count('.')
        
        # Domain length (phishing often uses longer domains)
        features[i, 5] = len(domain)
        
        # Path length
        features[i, 6] = len(path)
        
        # Path depth (count of directories)
        features[i, 7] = path.count('/')
        
        # Count of query parameters
        features[i, 8] = len(urllib.parse.parse_qs(parsed.query)) if hasattr(parsed, 'query') else 0
        
        # Presence of IP address in domain
        features[i, 9] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
        
        # Presence of hexadecimal characters in domain
        features[i, 10] = 1 if re.search(r'0x[0-9a-f]+', url_lower) else 0
        
        # Presence of suspicious TLDs
        features[i, 11] = 1 if any(tld in domain.lower() for tld in suspicious_tlds) else 0
        
        # Check for hyphens in domain (often used in phishing)
        features[i, 12] = domain.count('-')
        
        # Check for multiple subdomains (> 3 is suspicious)
        features[i, 13] = 1 if domain.count('.') > 2 else 0
        
        # Check for government indicators (both for legit detection and phishing targeting)
        has_gov_term = any(term in url_lower for term in government_terms)
        ends_with_gov = domain.lower().endswith('.gov')
        features[i, 14] = 1 if has_gov_term or ends_with_gov else 0
        
        # Check for e-commerce indicators
        has_ecom_brand = any(brand in url_lower for brand in ecommerce_brands)
        features[i, 15] = 1 if has_ecom_brand else 0
        
        # Check for brand names in domain but not at the start (e-commerce)
        for j, brand in enumerate(ecommerce_brands[:10]):  # Expanded to include all brands
            domain_parts = domain.lower().split('.')
            main_domain = domain_parts[0] if len(domain_parts) > 0 else ""
            
            # Brand exists in URL
            has_brand = brand in url_lower
            
            # Brand is in the main part of domain but not at start
            # This catches domains like "paypal-secure.com" or "secure-paypal.com"
            brand_in_domain_not_start = brand in main_domain and not main_domain.startswith(brand)
            
            # Brand is in domain but in a suspicious pattern (e.g., paypal-secure.com)
            brand_in_domain_suspicious = (f"{brand}-" in main_domain) or (f"-{brand}" in main_domain)
            
            # If brand exists but not as primary domain, likely phishing
            features[i, 16 + j] = 1 if (has_brand and (brand_in_domain_not_start or brand_in_domain_suspicious)) else 0
            
            # Additional check: Is brand mentioned in subdomain? (e.g. paypal.phishing.com)
            # This is a strong phishing indicator
            brand_in_subdomain = len(domain_parts) > 1 and any(brand in subdomain for subdomain in domain_parts[:-1])
            features[i, 26 + j] = 1 if brand_in_subdomain else 0
        
        # Presence of suspicious terms
        for j, term in enumerate(suspicious_terms[:8]):  # Using 8 terms, which is why we need more feature columns
            features[i, 36 + j] = 1 if term in url_lower else 0
    
    return sparse.csr_matrix(features)

custom_print("Extracting URL features...")
X_train_url_features_phish = extract_url_features(X_train_phish)
X_test_url_features_phish = extract_url_features(X_test_phish)

# Use same feature extraction for the category model
X_train_url_features_cat = extract_url_features(X_train_cat)
X_test_url_features_cat = extract_url_features(X_test_cat)

# Use same feature extraction for the combined model
X_train_url_features_combined = extract_url_features(X_train_combined)
X_test_url_features_combined = extract_url_features(X_test_combined)

# Combine features for each model
X_train_features_phish = sparse.hstack([X_train_tfidf_phish, X_train_url_features_phish])
X_test_features_phish = sparse.hstack([X_test_tfidf_phish, X_test_url_features_phish])

X_train_features_cat = sparse.hstack([X_train_tfidf_cat, X_train_url_features_cat])
X_test_features_cat = sparse.hstack([X_test_tfidf_cat, X_test_url_features_cat])

X_train_features_combined = sparse.hstack([X_train_tfidf_combined, X_train_url_features_combined])
X_test_features_combined = sparse.hstack([X_test_tfidf_combined, X_test_url_features_combined])

# -------- Train Models --------
# 1. Model for phishing detection (binary)
custom_print("CHECKPOINT: Starting phishing model training...")
phishing_model = XGBClassifier(
    learning_rate=0.05,
    max_depth=7,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.2,
    # Removed use_label_encoder parameter to fix warning
    eval_metric='logloss',
    scale_pos_weight=1.2,
    early_stopping_rounds=20 
)

eval_set_phish = [(X_train_features_phish, y_train_phish), (X_test_features_phish, y_test_phish)]

# Using a try-except block to catch potential errors
try:
    phishing_model.fit(
        X_train_features_phish, 
        y_train_phish, 
        eval_set=eval_set_phish,
        verbose=50  # Reduced verbosity - prints every 50 iterations instead of every iteration
    )
    custom_print("CHECKPOINT: Finished phishing model training.")
except Exception as e:
    custom_print(f"Error during phishing model training: {str(e)}")
    sys.exit(1)

# 2. Model for category prediction (gov vs ecommerce)
custom_print("CHECKPOINT: Starting category model training...")
category_model = XGBClassifier(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=150,
    subsample=0.8,
    colsample_bytree=0.8,
    # Removed use_label_encoder parameter to fix warning
    eval_metric='logloss'
)

eval_set_cat = [(X_train_features_cat, y_train_cat), (X_test_features_cat, y_test_cat)]
try:
    category_model.fit(
        X_train_features_cat, 
        y_train_cat, 
        eval_set=eval_set_cat,
        verbose=50  # Reduced verbosity
    )
    custom_print("CHECKPOINT: Finished category model training.")
except Exception as e:
    custom_print(f"Error during category model training: {str(e)}")
    sys.exit(1)

# 3. Combined model (single model for both tasks - 4 classes)
custom_print("CHECKPOINT: Starting combined model training...")
combined_model = XGBClassifier(
    learning_rate=0.05,
    max_depth=8,
    n_estimators=250,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.2,
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss'
)

eval_set_combined = [(X_train_features_combined, y_train_combined), (X_test_features_combined, y_test_combined)]
try:
    combined_model.fit(
        X_train_features_combined, 
        y_train_combined, 
        eval_set=eval_set_combined,
        verbose=50  # Reduced verbosity
    )
    custom_print("CHECKPOINT: Finished combined model training.")
except Exception as e:
    custom_print(f"Error during combined model training: {str(e)}")
    sys.exit(1)

# -------- Model Evaluation --------
# Add checkpoint to see if we reach evaluation
custom_print("CHECKPOINT: Starting model evaluation...")

# Evaluate phishing model
custom_print("\n===== PHISHING MODEL EVALUATION =====")
try:
    y_pred_phish = phishing_model.predict(X_test_features_phish)
    accuracy_phish = accuracy_score(y_test_phish, y_pred_phish)

    # Calculate confusion matrix and rates
    cm_phish = confusion_matrix(y_test_phish, y_pred_phish)
    tn, fp, fn, tp = cm_phish.ravel()

    # Calculate rates
    tpr = tp / (tp + fn)  # True positive rate (sensitivity, recall)
    fpr = fp / (fp + tn)  # False positive rate
    tnr = tn / (tn + fp)  # True negative rate (specificity)
    fnr = fn / (fn + tp)  # False negative rate

    custom_print(f"Accuracy: {accuracy_phish:.4f}")
    custom_print(f"True Positive Rate (sensitivity/recall): {tpr:.4f}")
    custom_print(f"False Positive Rate: {fpr:.4f}")
    custom_print(f"True Negative Rate (specificity): {tnr:.4f}")
    custom_print(f"False Negative Rate: {fnr:.4f}")
    custom_print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    custom_print("\nClassification Report:")
    custom_print(classification_report(y_test_phish, y_pred_phish))
except Exception as e:
    custom_print(f"Error during phishing model evaluation: {str(e)}")

# Evaluate category model
custom_print("\n===== CATEGORY MODEL EVALUATION =====")
try:
    y_pred_cat = category_model.predict(X_test_features_cat)
    accuracy_cat = accuracy_score(y_test_cat, y_pred_cat)

    cm_cat = confusion_matrix(y_test_cat, y_pred_cat)
    custom_print(f"Accuracy: {accuracy_cat:.4f}")
    custom_print(f"Confusion Matrix:")
    custom_print(cm_cat)
    custom_print("\nClassification Report:")
    custom_print(classification_report(y_test_cat, y_pred_cat, target_names=['Government', 'E-commerce']))
except Exception as e:
    custom_print(f"Error during category model evaluation: {str(e)}")

# Evaluate combined model
custom_print("\n===== COMBINED MODEL EVALUATION =====")
try:
    y_pred_combined = combined_model.predict(X_test_features_combined)
    accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)

    cm_combined = confusion_matrix(y_test_combined, y_pred_combined)
    custom_print(f"Accuracy: {accuracy_combined:.4f}")
    custom_print(f"Confusion Matrix:")
    custom_print(cm_combined)
    custom_print("\nClassification Report:")
    custom_print(classification_report(y_test_combined, y_pred_combined, 
                           target_names=['Legitimate Gov', 'Phishing Gov', 
                                         'Legitimate E-commerce', 'Phishing E-commerce']))
except Exception as e:
    custom_print(f"Error during combined model evaluation: {str(e)}")

custom_print("CHECKPOINT: Completed model evaluation.")

# -------- Save Models --------
custom_print("CHECKPOINT: Starting model saving...")
try:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    model_prefix = f"models/phishing_detector_v{timestamp}"

    # Save model artifacts
    joblib.dump(phishing_model, f"{model_prefix}_phishing_model.pkl")
    joblib.dump(category_model, f"{model_prefix}_category_model.pkl")
    joblib.dump(combined_model, f"{model_prefix}_combined_model.pkl")
    joblib.dump(tfidf, f"{model_prefix}_tfidf.pkl")

    # Save model metadata
    with open(f"{model_prefix}_metadata.txt", "w") as f:
        f.write(f"Models trained on: {now}\n")
        f.write(f"Dataset size: {len(df)}\n")
        f.write(f"Training set size: {len(X_train_phish)}\n")
        f.write(f"Test set size: {len(X_test_phish)}\n\n")
        
        f.write("=== PHISHING MODEL METRICS ===\n")
        f.write(f"Accuracy: {accuracy_phish:.4f}\n")
        f.write(f"True Positive Rate: {tpr:.4f}\n")
        f.write(f"False Positive Rate: {fpr:.4f}\n")
        f.write(f"True Negative Rate: {tnr:.4f}\n")
        f.write(f"False Negative Rate: {fnr:.4f}\n")
        f.write(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n\n")
        
        f.write("=== CATEGORY MODEL METRICS ===\n")
        f.write(f"Accuracy: {accuracy_cat:.4f}\n\n")
        
        f.write("=== COMBINED MODEL METRICS ===\n")
        f.write(f"Accuracy: {accuracy_combined:.4f}\n")

    custom_print(f"SUCCESS! Models and metadata saved with prefix: {model_prefix}")
except Exception as e:
    custom_print(f"Error during model saving: {str(e)}")

# -------- Comprehensive URL Checker Function --------
def check_url(url, phishing_model=phishing_model, category_model=category_model, 
              combined_model=combined_model, tfidf=tfidf):
    """
    Check if a URL is likely to be phishing or legitimate and determine its category
    
    Args:
        url (str): URL to check
        phishing_model: Model for phishing detection
        category_model: Model for category detection
        combined_model: Combined model for both tasks
        tfidf: Fitted TF-IDF vectorizer
        
    Returns:
        dict: Comprehensive results including predictions and analysis
    """
    # Extract features
    url_tfidf = tfidf.transform([url])
    url_features = extract_url_features([url])
    url_combined = sparse.hstack([url_tfidf, url_features])
    
    # Get URL parts for rule-based checks
    parsed = urllib.parse.urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
    domain = parsed.netloc if parsed.netloc else url.split('/')[0]
    domain_lower = domain.lower()
    
    # Make predictions with all models
    is_phishing = bool(phishing_model.predict(url_combined)[0])
    phishing_prob = float(phishing_model.predict_proba(url_combined)[0][1])
    
    # Get the actual predicted class as an integer
    category_class = int(category_model.predict(url_combined)[0])
    category_prob = float(category_model.predict_proba(url_combined)[0][category_class])
    category = "E-commerce" if category_class == 1 else "Government"
    
    combined_class = int(combined_model.predict(url_combined)[0])
    combined_probs = combined_model.predict_proba(url_combined)[0]
    
    # Map combined class to readable format
    combined_classes = ['Legitimate Government', 'Phishing Government', 
                        'Legitimate E-commerce', 'Phishing E-commerce']
    combined_result = combined_classes[combined_class]
    
    # Apply additional rule-based checks for common phishing patterns
    # This helps catch cases the model might miss
    
    # Rule 1: Brand name in domain but not at the start (e.g., secure-paypal.com)
    ecommerce_brands = ['paypal', 'amazon', 'ebay', 'walmart', 'alibaba', 'shopify', 
                     'netflix', 'etsy', 'bestbuy', 'target']
    
    # Check main domain part (before first dot)
    main_domain = domain_lower.split('.')[0]
    
    # Force phishing detection for cases where brand is in domain but not at start
    brand_in_domain_not_start = False
    detected_brand = None
    
    for brand in ecommerce_brands:
        if brand in main_domain and not main_domain.startswith(brand):
            brand_in_domain_not_start = True
            detected_brand = brand
            # Override model prediction
            is_phishing = True
            phishing_prob = max(0.85, phishing_prob)  # Set minimum 85% probability
            
            # If it was classified as e-commerce, update combined result
            if category == "E-commerce":
                combined_class = 3  # Phishing E-commerce
                combined_result = combined_classes[combined_class]
            break
    
    # Rule 2: Brand name with hyphens (e.g., paypal-secure.com, secure-paypal.com)
    if not brand_in_domain_not_start:
        for brand in ecommerce_brands:
            if (f"{brand}-" in main_domain or f"-{brand}" in main_domain):
                is_phishing = True
                phishing_prob = max(0.8, phishing_prob)
                detected_brand = brand
                
                # If e-commerce, update combined result
                if category == "E-commerce":
                    combined_class = 3  # Phishing E-commerce
                    combined_result = combined_classes[combined_class]
                break
    
    # Rule 3: Login/verification in URL for popular brands
    if any(brand in url.lower() for brand in ecommerce_brands) and \
       any(term in url.lower() for term in ['login', 'signin', 'secure', 'verify', 'account']):
        # Stronger indicator if it's not the official domain
        for brand in ecommerce_brands:
            official_domains = {
                'paypal': 'paypal.com',
                'amazon': 'amazon.com',
                'ebay': 'ebay.com',
                'walmart': 'walmart.com',
                'netflix': 'netflix.com'
            }
            
            if brand in url.lower() and brand in official_domains:
                if official_domains[brand] not in domain_lower:
                    is_phishing = True
                    phishing_prob = max(0.9, phishing_prob)
                    detected_brand = brand
                    
                    # If e-commerce, update combined result
                    if category == "E-commerce":
                        combined_class = 3  # Phishing E-commerce
                        combined_result = combined_classes[combined_class]
                    break
    
    # Rule 4: Government terms in non-government domains
    gov_terms = ['gov', 'government', 'federal', 'irs', 'tax', 'treasury', 'usps']
    if any(term in url.lower() for term in gov_terms) and not domain.lower().endswith('.gov'):
        is_phishing = True
        phishing_prob = max(0.85, phishing_prob)
        
        # If classified as government, update combined result
        if category == "Government":
            combined_class = 1  # Phishing Government
            combined_result = combined_classes[combined_class]
    
    # Determine risk level based on phishing probability
    risk_level = "High" if phishing_prob > 0.7 else "Medium" if phishing_prob > 0.4 else "Low"
    
    # Detailed analysis
    risk_signals = []
    
    # Add brand impersonation as a risk signal if detected
    if detected_brand:
        risk_signals.append(f"Brand impersonation detected: '{detected_brand}'")
    
    # Category-specific checks
    if category == "E-commerce":
        # Check common e-commerce brand name misuse (already done in rules above)
        # Add any additional e-commerce specific checks here
        for brand in ecommerce_brands:
            if brand in domain_lower and not domain_lower.startswith(brand) and not domain_lower.endswith(f".{brand}.com"):
                risk_signals.append(f"Potential e-commerce brand impersonation: '{brand}' in domain but not positioned correctly")
                break
    else:  # Government
        # Check for government site impersonation
        if not domain_lower.endswith('.gov') and any(term in url.lower() for term in ['gov', 'government', 'federal', 'irs']):
            risk_signals.append("Potential government site impersonation: Uses government terms but not a .gov domain")
    
    # General checks
    # Check for hyphen usage (common in phishing)
    if domain.count('-') > 1:
        risk_signals.append(f"Multiple hyphens in domain ({domain.count('-')})")
    
    # Check for suspicious TLDs
    suspicious_tlds = ['.xyz', '.top', '.club', '.online', '.site', '.info']
    for tld in suspicious_tlds:
        if domain_lower.endswith(tld):
            risk_signals.append(f"Suspicious TLD: {tld}")
            break
    
    # Check for suspicious terms
    suspicious_terms = ['login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm', 'support']
    found_terms = [term for term in suspicious_terms if term in url.lower()]
    if found_terms:
        risk_signals.append(f"Suspicious terms: {', '.join(found_terms)}")
    
    # Check for excessive subdomains
    if domain.count('.') > 2:
        risk_signals.append(f"Excessive subdomains ({domain.count('.')})")
    
    # Prepare comprehensive result
    result = {
        "url": url,
        "is_phishing": is_phishing,
        "phishing_probability": phishing_prob,
        "category": category,
        "category_probability": category_prob,
        "combined_result": combined_result,
        "combined_probabilities": {
            combined_classes[i]: float(combined_probs[i]) for i in range(4)
        },
        "risk_level": risk_level,
        "risk_signals": risk_signals
    }
    
    return result

# Demo the URL checker with examples
custom_print("\n===== COMPREHENSIVE URL CHECKER DEMO =====")
test_urls = [
    "https://www.amazon.com",
    "https://www.amaz0n-secure-login.com",
    "https://www.usa.gov",
    "http://verify-gov-account.com",
    "https://www.paypal-support.com/login",  # The problematic PayPal URL
    "https://www.paypal.com/login",          # Legitimate PayPal
    "https://irs-tax-refund.com",            # Government phishing example
    "https://www.irs.gov/refunds"            # Legitimate government
]

for url in test_urls:
    result = check_url(url)
    print(f"\nURL: {result['url']}")
    print(f"Combined Result: {result['combined_result']}")
    print(f"Phishing: {'YES' if result['is_phishing'] else 'NO'} (Probability: {result['phishing_probability']:.4f})")
    print(f"Category: {result['category']} (Probability: {result['category_probability']:.4f})")
    print(f"Risk Level: {result['risk_level']}")
    
    if result['risk_signals']:
        print("Risk signals detected:")
        for signal in result['risk_signals']:
            print(f"  - {signal}")
    
    print("---")