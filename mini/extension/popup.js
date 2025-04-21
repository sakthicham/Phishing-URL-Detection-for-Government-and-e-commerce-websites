document.addEventListener('DOMContentLoaded', function () {
  addRippleEffect();
  
  // Get references to UI elements
  const statusEl = document.getElementById("phish-status");
  const loaderEl = document.getElementById("loader");
  const warningBox = document.getElementById("warning");
  const status = document.getElementById("status");
  const containerEl = document.querySelector('.container');
  
  // Set initial state
  if (statusEl) {
    statusEl.textContent = "DETECTION IN PROGRESS...";
    statusEl.className = "analyzing";
    statusEl.style.display = "block";
  }
  if (loaderEl) {
    loaderEl.style.display = "block";
  }

  // Get current URL
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const url = tabs[0].url;
    const urlDisplay = document.getElementById("url-display");
    
    // Display URL with typewriter effect
    if (urlDisplay) {
      typeWriter(urlDisplay, url, 50);
    }
    
    // Check if URL is phishing
    processUrl(url, tabs[0].id);
  });
  
  // Main URL processing function
  function processUrl(url, tabId) {
    console.log("Processing URL:", url);
    
    try {
      const hostname = new URL(url).hostname.toLowerCase();
      
      // Handle PayPal domains
      if (hostname.includes("paypal")) {
        if (hostname === "paypal.com" || hostname === "www.paypal.com") {
          showSafeResult("PayPal");
        } else {
          showPhishingResult("PayPal");
        }
        return;
      }
      
      // Handle Flipkart domains
      if (hostname.includes("flipkart")) {
        if (hostname === "flipkart.com" || hostname === "www.flipkart.com") {
          showSafeResult("Flipkart");
        } else {
          showPhishingResult("Flipkart");
        }
        return;
      }
      
      // Handle Amazon domains
      if (hostname.includes("amazon")) {
        const legitAmazon = ["amazon.com", "www.amazon.com", "amazon.in", "www.amazon.in"];
        if (legitAmazon.includes(hostname)) {
          showSafeResult("Amazon");
        } else {
          showPhishingResult("Amazon");
        }
        return;
      }
      
      // For all other URLs, try with backend
      checkWithBackend(url, tabId);
      
    } catch (e) {
      console.error("Error processing URL:", e);
      checkWithBackend(url, tabId);
    }
  }
  
  // Function to handle phishing detection
  function showPhishingResult(brand) {
    console.log("DETECTED: Phishing " + brand + " site");
    
    // Update status element
    if (statusEl) {
      statusEl.textContent = "⚠️ DETECTED: Phishing " + brand + " site";
      statusEl.className = "phishing";
      statusEl.style.display = "block";
    }
    
    // Update warning box
    if (warningBox) {
      warningBox.style.display = "block";
    }
    
    // Update status message
    if (status) {
      status.textContent = `⚠️ Potential ${brand} phishing site`;
    }
    
    // Add animation class
    if (containerEl) {
      containerEl.classList.add("phishing-anim");
    }
    
    // Hide loader
    if (loaderEl) {
      loaderEl.style.display = "none";
    }
  }
  
  // Function to handle safe sites
  function showSafeResult(brand) {
    console.log("DETECTED: Legitimate " + brand + " site");
    
    // Update status element
    if (statusEl) {
      statusEl.textContent = "✅ DETECTED: Legitimate " + brand + " Website";
      statusEl.className = "safe";
      statusEl.style.display = "block";
    }
    
    // Hide warning box
    if (warningBox) {
      warningBox.style.display = "none";
    }
    
    // Update status message
    if (status) {
      status.textContent = "✅ This website looks safe.";
    }
    
    // Add animation class
    if (containerEl) {
      containerEl.classList.add("safe-anim");
    }
    
    // Hide loader
    if (loaderEl) {
      loaderEl.style.display = "none";
    }
  }
  
  // Function to check with backend
  function checkWithBackend(url, tabId) {
    console.log("Checking with backend:", url);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      console.error("Backend request timed out");
      
      // Default to a neutral message if backend check fails
      if (statusEl) {
        statusEl.textContent = "DETECTION COMPLETE: Unable to verify";
        statusEl.className = "analyzing";
        statusEl.style.display = "block";
      }
      
      if (loaderEl) {
        loaderEl.style.display = "none";
      }
      
    }, 5000); // 5 second timeout

    fetch("http://127.0.0.1:5000/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url }),
      signal: controller.signal
    })
      .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
      })
      .then(data => {
        console.log("Backend response:", data);
        
        if (data.is_phishing) {
          const brand = data.url_type === "government" ? "Government" : "E-Commerce";
          showPhishingResult(brand);
        } else {
          const brand = data.url_type === "government" ? "Government" : "E-Commerce";
          showSafeResult(brand);
        }
      })
      .catch(error => {
        clearTimeout(timeoutId);
        console.error("Backend error:", error);
        
        // Show a clear error message
        if (statusEl) {
          statusEl.textContent = "DETECTION ERROR: Please try again";
          statusEl.className = "phishing";
          statusEl.style.display = "block";
        }
        
        if (loaderEl) {
          loaderEl.style.display = "none";
        }
      });
  }

  function typeWriter(element, text, speed) {
    let i = 0;
    element.textContent = "";
    const adjustedSpeed = text.length > 30 ? 10 : speed;
    setTimeout(function type() {
      if (i < text.length) {
        element.textContent += text.charAt(i++);
        setTimeout(type, adjustedSpeed);
      }
    }, adjustedSpeed);
  }

  function addRippleEffect() {
    const container = document.querySelector('.container');
    const centerX = container.offsetWidth / 2;
    const centerY = container.offsetHeight / 2;

    const ripple = document.createElement('div');
    ripple.className = 'ripple';
    ripple.style.left = (centerX - 32) + 'px';
    ripple.style.top = (centerY - 32) + 'px';
    container.appendChild(ripple);

    setInterval(() => {
      const oldRipple = document.querySelector('.ripple');
      if (oldRipple) {
        const newRipple = document.createElement('div');
        newRipple.className = 'ripple';
        newRipple.style.left = (centerX - 32) + 'px';
        newRipple.style.top = (centerY - 32) + 'px';
        container.appendChild(newRipple);

        setTimeout(() => {
          if (oldRipple && oldRipple.parentNode) {
            oldRipple.parentNode.removeChild(oldRipple);
          }
        }, 1500);
      }
    }, 3000);
  }
});