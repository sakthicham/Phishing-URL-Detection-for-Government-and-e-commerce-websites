// PhishGuard content script
console.log("PhishGuard content script loaded for automatic detection");

// Function to check if URL is internal browser page
function isInternalUrl(url) {
  return /^(chrome|chrome-extension|about|file):/.test(url);
}

// Only proceed if we're on a web page (not an internal browser page)
if (!isInternalUrl(window.location.href)) {
  // Send the current URL to the background script for checking
  chrome.runtime.sendMessage({
    action: "checkURL",
    url: window.location.href
  }, response => {
    console.log("Got response from background script:", response);
  });

  // Also listen for DOM changes that might indicate a single-page app navigation
  const observer = new MutationObserver(mutations => {
    // Check if URL has changed (for single-page apps)
    if (window.location.href !== lastCheckedUrl) {
      lastCheckedUrl = window.location.href;
      
      // Send the new URL to the background script
      chrome.runtime.sendMessage({
        action: "checkURL",
        url: window.location.href
      });
    }
  });

  // Track the last checked URL to avoid duplicate checks
  let lastCheckedUrl = window.location.href;

  // Start observing changes to the DOM
  observer.observe(document, {
    subtree: true,
    childList: true
  });

  // Also check on history changes (browser back/forward, etc.)
  window.addEventListener('popstate', () => {
    // Send the new URL to the background script
    chrome.runtime.sendMessage({
      action: "checkURL",
      url: window.location.href
    });
  });
}