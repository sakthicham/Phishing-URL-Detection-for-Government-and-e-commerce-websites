// Background script for automatic phishing detection
console.log("PhishGuard background script initialized");

// Keep service worker alive
const keepAlive = () => setInterval(chrome.runtime.getPlatformInfo, 20e3);
chrome.runtime.onStartup.addListener(keepAlive);
chrome.runtime.onInstalled.addListener(keepAlive);

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "checkURL") {
    console.log("Content script requested URL check for:", request.url);
    checkUrlWithBackend(request.url, sender.tab.id);
    sendResponse({status: "checking"});
  }
  return true; // Keep the message channel open for async responses
});

// Function to check URL with backend server
function checkUrlWithBackend(url, tabId) {
  console.log(`Checking URL in background: ${url}`);
  
  fetch("http://127.0.0.1:5000/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: url }),
    signal: AbortSignal.timeout(5000) // 5 second timeout
  })
    .then(response => {
      if (!response.ok) throw new Error('Network response was not ok');
      return response.json();
    })
    .then(data => {
      console.log("Backend response:", data);
      
      if (data.is_phishing) {
        // Show warning for phishing site
        const brand = data.url_type === "government" ? "Government" : "E-Commerce";
        showPhishingWarning(tabId, brand, data);
      }
    })
    .catch(error => {
      console.error("Backend error:", error);
    });
}

// Function to show phishing warning
function showPhishingWarning(tabId, brand, data) {
  // First, inject our CSS for the warning banner
  chrome.scripting.insertCSS({
    target: { tabId: tabId },
    files: ['warning-banner.css']
  }).then(() => {
    // Then inject and execute the warning banner code
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      function: injectWarningBanner,
      args: [brand, data]
    });
    
    // Also show a Chrome notification
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/warning-128.png',
      title: '⚠️ Phishing Alert!',
      message: `This ${brand} website appears to be a phishing attempt. Be careful!`,
      priority: 2
    });
  }).catch(err => console.error("Error injecting warning:", err));
}

// This function runs in the context of the page
function injectWarningBanner(brand, data) {
  // Check if banner already exists
  if (document.getElementById('phishing-warning-banner')) return;
  
  // Create warning banner
  const banner = document.createElement('div');
  banner.id = 'phishing-warning-banner';
  banner.innerHTML = `
    <div class="phishing-warning-content">
      <div class="phishing-warning-icon">⚠️</div>
      <div class="phishing-warning-text">
        <strong>PHISHING ALERT!</strong> 
        <p>This ${brand} website appears to be fake. Risk level: ${data.risk_level}</p>
      </div>
      <div class="phishing-warning-actions">
        <button id="phishing-warning-close">Dismiss</button>
        <button id="phishing-warning-leave">Leave Site</button>
      </div>
    </div>
  `;
  
  // Add to page
  document.body.insertBefore(banner, document.body.firstChild);
  
  // Add event listeners
  document.getElementById('phishing-warning-close').addEventListener('click', function() {
    banner.style.display = 'none';
  });
  
  document.getElementById('phishing-warning-leave').addEventListener('click', function() {
    window.location.href = 'about:blank';
  });
}

// Listen for navigation events
chrome.webNavigation.onCompleted.addListener(function(details) {
  // Only check main frame (not iframes, etc)
  if (details.frameId === 0) {
    // Add a small delay to ensure page is fully loaded
    setTimeout(() => {
      checkUrlWithBackend(details.url, details.tabId);
    }, 1000);
  }
});

// Also check when tabs are updated
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
  // Only check when loading is complete and URL is changing
  if (changeInfo.status === 'complete' && tab.url) {
    // Avoid checking chrome:// and other internal URLs
    if (!/^(chrome|chrome-extension|about|file):/.test(tab.url)) {
      checkUrlWithBackend(tab.url, tabId);
    }
  }
});