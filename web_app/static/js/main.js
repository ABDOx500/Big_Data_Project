/**
 * Spam Detection System - Frontend JavaScript
 * 
 * Handles all client-side interactions including:
 * - Form submission and prediction requests
 * - Real-time statistics updates
 * - Prediction history display
 * - Model information loading
 * - Error handling and user feedback
 * 
 * Author: Big Data Project
 * Date: January 2025
 */

// ============================================
// Configuration
// ============================================
const API_BASE_URL = window.location.origin;

const EXAMPLE_MESSAGES = {
    spam: "URGENT! You have won a $1,000 Walmart gift card. Click here now to claim your prize! Limited time offer. Text CLAIM to 12345.",
    ham: "Hey! Are you free for lunch tomorrow? I was thinking we could try that new Italian restaurant downtown. Let me know!"
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    // Form elements
    form: document.getElementById('prediction-form'),
    messageInput: document.getElementById('message-input'),

    predictBtn: document.getElementById('predict-btn'),
    btnText: document.querySelector('.btn-text'),
    btnLoader: document.querySelector('.btn-loader'),
    charCount: document.getElementById('char-count'),

    // Result elements
    resultPlaceholder: document.getElementById('result-placeholder'),
    resultDisplay: document.getElementById('result-display'),
    errorDisplay: document.getElementById('error-display'),
    resultLabel: document.getElementById('result-label'),
    confidenceValue: document.getElementById('confidence-value'),
    confidenceFill: document.getElementById('confidence-fill'),
    modelUsed: document.getElementById('model-used'),
    processingTime: document.getElementById('processing-time'),
    timestamp: document.getElementById('timestamp'),
    messagePreview: document.getElementById('message-preview'),
    errorMessage: document.getElementById('error-message'),

    // Statistics elements
    totalPredictions: document.getElementById('total-predictions'),
    spamCount: document.getElementById('spam-count'),
    hamCount: document.getElementById('ham-count'),
    spamPercentage: document.getElementById('spam-percentage'),

    // History elements
    historyTbody: document.getElementById('history-tbody'),
    refreshHistoryBtn: document.getElementById('refresh-history'),


    // Example buttons
    exampleButtons: document.querySelectorAll('.btn-example')
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Spam Detection System initialized');

    // Set up event listeners
    setupEventListeners();

    // Load initial data
    loadStatistics();
    loadHistory();


    // Update character count
    updateCharCount();
});

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    // Form submission
    elements.form.addEventListener('submit', handleFormSubmit);

    // Character counter
    elements.messageInput.addEventListener('input', updateCharCount);

    // Example buttons
    elements.exampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const exampleType = btn.dataset.example;
            elements.messageInput.value = EXAMPLE_MESSAGES[exampleType];
            updateCharCount();
            elements.messageInput.focus();
        });
    });

    // Refresh history button
    elements.refreshHistoryBtn.addEventListener('click', loadHistory);
}

// ============================================
// Form Handling
// ============================================
async function handleFormSubmit(event) {
    event.preventDefault();

    const message = elements.messageInput.value.trim();

    if (!message) {
        showError('Please enter a message to analyze');
        return;
    }

    // Show loading state
    setLoadingState(true);
    hideError();
    hideResult();

    try {
        // Make prediction request
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const result = await response.json();

        // Display result
        displayResult(result, message);

        // Update statistics and history
        loadStatistics();
        loadHistory();

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to get prediction. Please try again.');
    } finally {
        setLoadingState(false);
    }
}

// ============================================
// Result Display
// ============================================
function displayResult(result, originalMessage) {
    // Hide placeholder and error
    elements.resultPlaceholder.style.display = 'none';
    elements.errorDisplay.style.display = 'none';

    // Show result display
    elements.resultDisplay.style.display = 'block';

    // Update prediction label
    const isSpam = result.prediction.toLowerCase() === 'spam';
    elements.resultLabel.textContent = result.prediction.toUpperCase();
    elements.resultLabel.className = `result-label ${result.prediction.toLowerCase()}`;

    // Update confidence
    const confidencePercent = (result.confidence * 100).toFixed(1);
    elements.confidenceValue.textContent = `${confidencePercent}%`;
    elements.confidenceFill.style.width = `${confidencePercent}%`;

    // Update details
    elements.modelUsed.textContent = formatModelName(result.model_used);
    elements.processingTime.textContent = `${result.processing_time_ms} ms`;
    elements.timestamp.textContent = formatTimestamp(result.timestamp);

    // Update message preview
    elements.messagePreview.textContent = originalMessage;

    // Animate result
    elements.resultDisplay.style.animation = 'none';
    setTimeout(() => {
        elements.resultDisplay.style.animation = 'fadeIn 0.3s ease-in-out';
    }, 10);
}

function hideResult() {
    elements.resultDisplay.style.display = 'none';
}

// ============================================
// Error Handling
// ============================================
function showError(message) {
    elements.resultPlaceholder.style.display = 'none';
    elements.resultDisplay.style.display = 'none';
    elements.errorDisplay.style.display = 'block';
    elements.errorMessage.textContent = message;
}

function hideError() {
    elements.errorDisplay.style.display = 'none';
}

// ============================================
// Loading State
// ============================================
function setLoadingState(isLoading) {
    elements.predictBtn.disabled = isLoading;

    if (isLoading) {
        elements.btnText.style.display = 'none';
        elements.btnLoader.style.display = 'flex';
    } else {
        elements.btnText.style.display = 'block';
        elements.btnLoader.style.display = 'none';
    }
}

// ============================================
// Statistics
// ============================================
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);

        if (!response.ok) {
            throw new Error('Failed to load statistics');
        }

        const stats = await response.json();

        // Update statistics display with animation
        animateValue(elements.totalPredictions, stats.total_predictions);
        animateValue(elements.spamCount, stats.spam_count);
        animateValue(elements.hamCount, stats.ham_count);
        elements.spamPercentage.textContent = `${stats.spam_percentage}%`;



    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// ============================================
// History
// ============================================
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=10`);

        if (!response.ok) {
            throw new Error('Failed to load history');
        }

        const data = await response.json();

        // Clear existing rows
        elements.historyTbody.innerHTML = '';

        if (data.history.length === 0) {
            elements.historyTbody.innerHTML = `
                <tr class="empty-state">
                    <td colspan="4">No predictions yet. Start analyzing messages!</td>
                </tr>
            `;
            return;
        }

        // Add history rows
        data.history.forEach(item => {
            const row = createHistoryRow(item);
            elements.historyTbody.appendChild(row);
        });

    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function createHistoryRow(item) {
    const row = document.createElement('tr');

    const messageCell = document.createElement('td');
    messageCell.textContent = truncateText(item.message, 60);

    const predictionCell = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `prediction-badge ${item.prediction.toLowerCase()}`;
    badge.textContent = item.prediction.toUpperCase();
    predictionCell.appendChild(badge);

    const confidenceCell = document.createElement('td');
    confidenceCell.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    const timeCell = document.createElement('td');
    timeCell.textContent = formatTimestamp(item.timestamp);

    row.appendChild(messageCell);
    row.appendChild(predictionCell);
    row.appendChild(confidenceCell);
    row.appendChild(timeCell);

    return row;
}




// ============================================
// Utility Functions
// ============================================
function updateCharCount() {
    const count = elements.messageInput.value.length;
    elements.charCount.textContent = count;
}

function formatModelName(modelName) {
    const names = {
        'naive_bayes': 'Naive Bayes',
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest'
    };
    return names[modelName] || modelName;
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;

    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function animateValue(element, targetValue) {
    const currentValue = parseInt(element.textContent) || 0;
    const duration = 500; // ms
    const steps = 20;
    const stepValue = (targetValue - currentValue) / steps;
    const stepDuration = duration / steps;

    let currentStep = 0;

    const interval = setInterval(() => {
        currentStep++;
        const newValue = Math.round(currentValue + (stepValue * currentStep));
        element.textContent = newValue;

        if (currentStep >= steps) {
            element.textContent = targetValue;
            clearInterval(interval);
        }
    }, stepDuration);
}

// ============================================
// Auto-refresh (Optional)
// ============================================
// Uncomment to enable auto-refresh of statistics every 30 seconds
// setInterval(() => {
//     loadStatistics();
//     loadHistory();
// }, 30000);

// ============================================
// Console Welcome Message
// ============================================
console.log('%c🛡️ Spam Detection System', 'font-size: 20px; font-weight: bold; color: #6366f1;');
console.log('%cBuilt with Apache Spark ML & Flask', 'font-size: 12px; color: #6b7280;');
console.log('%cAPI Base URL:', 'font-weight: bold;', API_BASE_URL);
