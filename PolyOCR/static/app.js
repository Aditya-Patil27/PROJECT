// PolyOCR Additional JavaScript Functions

// Add smooth scrolling to results
function scrollToResults() {
    const resultsElement = document.getElementById('results');
    if (resultsElement) {
        resultsElement.scrollIntoView({ behavior: 'smooth' });
    }
}

// Add copy to clipboard functionality
function copyToClipboard(text, elementId) {
    navigator.clipboard.writeText(text).then(function() {
        // Show success message
        const element = document.getElementById(elementId);
        if (element) {
            const originalHTML = element.innerHTML;
            element.innerHTML = '<i class="fas fa-check"></i> Copied!';
            element.classList.add('text-success');
            
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.classList.remove('text-success');
            }, 2000);
        }
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
    });
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+U or Cmd+U to trigger file upload
    if ((event.ctrlKey || event.metaKey) && event.key === 'u') {
        event.preventDefault();
        document.getElementById('imageInput').click();
    }
    
    // Enter to analyze image (if file is selected)
    if (event.key === 'Enter' && !document.getElementById('analyzeBtn').disabled) {
        event.preventDefault();
        document.getElementById('analyzeBtn').click();
    }
});

// Add progress enhancement
function enhanceProgressFeedback() {
    const progressMessages = [
        "Loading AI models...",
        "Processing image...",
        "Extracting text...",
        "Detecting language...",
        "Applying corrections...",
        "Finalizing results..."
    ];
    
    let messageIndex = 0;
    const loadingText = document.querySelector('#loading p');
    
    const interval = setInterval(() => {
        if (loadingText && document.getElementById('loading').style.display !== 'none') {
            loadingText.textContent = progressMessages[messageIndex];
            messageIndex = (messageIndex + 1) % progressMessages.length;
        } else {
            clearInterval(interval);
        }
    }, 2000);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('PolyOCR initialized successfully');
    
    // Add fade-in animation to main container
    const mainContainer = document.querySelector('.main-container');
    if (mainContainer) {
        mainContainer.classList.add('fade-in');
    }
});
