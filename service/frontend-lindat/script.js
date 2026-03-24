/**
 * Category descriptions mapping to ensure labels are descriptive
 * even if the backend only returns the raw label code.
 */
const CATEGORY_DESCRIPTIONS = {
    "TEXT": "📰 Mixed Text: Mixtures of printed, handwritten, or typed text.",
    "TEXT_T": "📄 Typed: Machine-typed paragraphs.",
    "TEXT_P": "📄 Printed: Published paragraphs.",
    "TEXT_HW": "✏️📄 Handwritten: Handwritten paragraphs.",
    "LINE_T": "📏 Typed Table: Machine-typed tabular structure.",
    "LINE_P": "📏 Printed Table: Printed tabular structure.",
    "LINE_HW": "✏️📏 Handwritten Table: Handwritten tabular structure.",
    "DRAW": "📈 Drawing: Maps, paintings, schematics.",
    "DRAW_L": "📈📏 Structured Drawing: Drawings in a layout/legend.",
    "PHOTO": "🌄 Photo: Photographs.",
    "PHOTO_L": "🌄📏 Structured Photo: Photos in a table layout."
};

/**
 * Renders the prediction results into the DOM.
 * Handles both single image responses and multipage PDF responses.
 */
function renderResults(data) {
    const container = document.getElementById('results');
    container.innerHTML = '';

    if (!data) {
        container.innerHTML = `<div class="alert alert-warning">No data returned from server.</div>`;
        return;
    }

    if (data.error) {
         container.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
         return;
    }

    // --- Scenario 1: PDF Document (Multiple Pages) ---
    if (data.type === 'document') {
        let html = `<div style="margin-bottom: 1rem;">`;
        html += `<h2>Results for ${data.filename}</h2>`;
        html += `<p class="text-muted">Model: ${data.model_version}</p>`;
        html += `</div>`;

        if (!data.pages || data.pages.length === 0) {
            html += `<div class="alert alert-warning">No pages found in document.</div>`;
        } else {
            data.pages.forEach(page => {
                html += `<div class="model-card">`;
                html += `<div class="result-row">`; // Flex container

                // 1. Thumbnail Column (From Backend Base64)
                html += `<div class="thumb-col">`;
                if (page.thumbnail) {
                    html += `<img src="data:image/png;base64,${page.thumbnail}" alt="Page ${page.page} Thumbnail">`;
                } else {
                    html += `<span class="text-muted">No Preview</span>`;
                }
                html += `</div>`;

                // 2. Data Column
                html += `<div class="data-col">`;
                html += `<h4>Page ${page.page}</h4>`;
                html += renderPredictionsList(page.predictions);
                html += `</div>`;

                html += `</div>`; // End result-row
                html += `</div>`; // End model-card
            });
        }
        container.innerHTML = html;
        return;
    }

    // --- Scenario 2: Single Image ---
    // We get the local file object to create a preview, as the backend usually just sends JSON
    const fileInput = document.getElementById('imageInput');
    const localFile = fileInput.files.length > 0 ? fileInput.files[0] : null;
    const thumbSrc = localFile ? URL.createObjectURL(localFile) : null;

    let html = `<div class="model-card">`;
    html += `<div class="result-row">`; // Flex container

    // 1. Thumbnail Column (From Local File API)
    html += `<div class="thumb-col">`;
    if (thumbSrc) {
        html += `<img src="${thumbSrc}" alt="Uploaded Image Preview">`;
    } else {
        html += `<span class="text-muted">No Preview</span>`;
    }
    html += `</div>`;

    // 2. Data Column
    html += `<div class="data-col">`;
    html += `<h3>${data.model_version || 'Prediction Results'}</h3>`;
    if (data.predictions) {
        html += renderPredictionsList(data.predictions);
    } else {
        html += `<div class="alert alert-warning">No predictions returned.</div>`;
    }
    html += `</div>`;

    html += `</div>`; // End result-row
    html += `</div>`; // End model-card

    container.innerHTML = html;
}

/**
 * Helper to generate the list of bars for a set of predictions.
 * Applies descriptions from backend or fallback map.
 */
function renderPredictionsList(predictions) {
    if (!predictions || !predictions.length) return '<p>No scores available.</p>';

    let html = '';
    predictions.forEach(item => {
        const scorePct = (item.score * 100).toFixed(2);
        // Dynamic color for score bar (Green > 90%, Yellow > 50%, Red < 50%)
        const color = item.score > 0.9 ? '#4CAF50' : (item.score > 0.5 ? '#FFC107' : '#F44336');

        // Logic: Use description from backend if exists, otherwise lookup in constant
        const descriptionText = item.description || CATEGORY_DESCRIPTIONS[item.label] || "";
        const descHtml = descriptionText ? `<br><small class="text-muted">${descriptionText}</small>` : '';

        html += `
            <div style="margin-bottom: 0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items: flex-end;">
                    <span><strong>${item.label}</strong>${descHtml}</span>
                    <span style="font-weight:bold;">${scorePct}%</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill" style="width: ${scorePct}%; background-color: ${color};"></div>
                </div>
            </div>
        `;
    });
    return html;
}

// --- Tab Switching Logic ---
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            // Add active to current
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
}

// --- Legacy LINDAT Helper Functions ---
function updateLoadedStyles() {
  $('#loaded-styles').empty();
  $('.css-select').prop('disabled', false);
  $('link[rel="stylesheet"]').each(function () {
    $('#loaded-styles').append($(this).attr('href') + '\n');
  });
}

function injectStylesheets(url) {
  var SERVICE_URL = 'https://lindat-extractor.herokuapp.com';
  localStorage.setItem('url', url);
  $('link[injected]').remove();

  if (!url) {
    updateLoadedStyles();
    return;
  }

  $.ajax(SERVICE_URL + '/styles', {
    data: { uri: url }
  }).done(function (data) {
    data.forEach(function (item) {
      if (/lindat\.css$/.test(item)) return;
      $('head').append('<link rel="stylesheet" href="'+ item + '" type="text/css" injected="injected" />');
    });
    updateLoadedStyles();
  });
}

function switchHandle(handle, title) {
  var refbox = $('#refbox');
  if (refbox.lindatRefBox) {
      localStorage.setItem('handle', handle);
      localStorage.setItem('title', title);
      refbox.attr('handle', handle);
      refbox.attr('title', title);
      refbox.lindatRefBox({
        rest: 'https://lindat.mff.cuni.cz/repository/rest'
      });
  }
}

// --- Main Initialization ---
function init() {
    console.log("Initializing App...");

    // Initialize Tabs
    initTabs();

    // File Input UI enhancement
    const fileInput = document.getElementById('imageInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                fileNameDisplay.textContent = "Selected: " + e.target.files[0].name;
            } else {
                fileNameDisplay.textContent = "";
            }
        });
    }

    // Setup Form Handling
    const form = document.getElementById('classifyForm');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const resultDiv = document.getElementById('results');
            const loader = document.getElementById('loading');

            // Check file presence
            if (!fileInput.files.length) {
                alert("Please select a file first.");
                return;
            }

            const file = fileInput.files[0];

            // 1. Determine Correct Endpoint
            const isPdf = file.type === 'application/pdf';
            const endpointPath = isPdf ? '/predict_document' : '/predict_image';

            // --- REFINED: Derive base URL from the current origin dynamically ---
            // Assumes API is hosted alongside the frontend, or uses a relative path
            let baseUrl = window.location.origin;

            // If the user manually provided a META tag for cross-origin APIs (optional architecture best practice)
            const metaApiUrl = document.querySelector('meta[name="api-base-url"]');
            if (metaApiUrl) {
                baseUrl = metaApiUrl.getAttribute('content');
            } else if (window.location.port === '8080') {
                 // Dev fallback
                 baseUrl = 'http://localhost:8000';
            }
            const finalUrl = baseUrl + endpointPath;

            // UI Reset
            resultDiv.innerHTML = '';
            loader.style.display = 'block';

            const formData = new FormData(e.target);

            try {
                const response = await fetch(finalUrl, {
                    method: 'POST',
                    body: formData
                });

                // Robust Content-Type Check to prevent JSON parse errors
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.indexOf("application/json") === -1) {
                    const text = await response.text();
                    // If we get HTML, it's likely a 404 from the dev server or proxy error
                    if (text.trim().startsWith('<')) {
                        throw new Error(`Server returned HTML instead of JSON. Ensure API is running at ${baseUrl || 'current origin'} and endpoints are correct.`);
                    }
                    throw new Error(`Invalid response type: ${contentType}`);
                }

                if (!response.ok) {
                    const errJson = await response.json();
                    throw new Error(errJson.detail || `Server Error: ${response.statusText}`);
                }

                const data = await response.json();
                renderResults(data);

            } catch (err) {
                console.error(err);
                resultDiv.innerHTML = `<div class="model-card" style="border-color: #F44336; color: #721c24; background-color: #f8d7da;">
                    <strong>Error:</strong> ${err.message}
                </div>`;
            } finally {
                loader.style.display = 'none';
            }
        });
    } else {
        console.error("Form element 'classifyForm' not found!");
    }

    // Legacy LINDAT settings
    var lang = localStorage.getItem('lang') || 'en';
    var project = localStorage.getItem('project') || 'lindat-home';
    var handle = localStorage.getItem('handle') || '11234/1-1508';
    var title = localStorage.getItem('title') || 'HamleDT 3.0';
    var url  = localStorage.getItem('url') || '';

    try {
        injectStylesheets(url);
        switchHandle(handle, title);
    } catch(e) {
        console.warn("LINDAT stylesheet injection failed:", e);
    }

    document.body.setAttribute('id', project);
}

// Execute on page load using jQuery
$(init);