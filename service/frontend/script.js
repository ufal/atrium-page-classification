/**
 * Renders the prediction results into the DOM.
 * Expects new data structure:
 * {
 * "model_version": "...",
 * "predictions": [ { "label": "A", "score": 0.9 }, ... ]
 * }
 */
function renderResults(data) {
    const container = document.getElementById('results');
    container.innerHTML = '';

    if (!data || !data.predictions) {
        container.innerHTML = `<div class="alert alert-warning">No predictions returned from server.</div>`;
        return;
    }

    // New format returns a single object with metadata and a list of predictions
    // This is true even for 'all' now (which is averaged).

    let html = `<div class="model-card">`;
    html += `<h3>${data.model_version}</h3>`;

    const items = data.predictions;

    items.forEach(item => {
        const scorePct = (item.score * 100).toFixed(2);
        // Dynamic color for score bar (Green > 90%, Yellow > 50%, Red < 50%)
        const color = item.score > 0.9 ? '#4CAF50' : (item.score > 0.5 ? '#FFC107' : '#F44336');

        html += `
            <div style="margin-bottom: 0.5rem;">
                <div style="display:flex; justify-content:space-between;">
                    <span><strong>${item.label}</strong></span>
                    <span>${scorePct}%</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill" style="width: ${scorePct}%; background-color: ${color};"></div>
                </div>
            </div>
        `;
    });
    html += `</div>`;

    container.innerHTML = html;
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
            document.getElementById(targetId).classList.add('active');
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
  // Only proceed if refbox plugin exists
  if (!refbox.lindatRefBox) return;

  localStorage.setItem('handle', handle);
  localStorage.setItem('title', title);
  refbox.attr('handle', handle);
  refbox.attr('title', title);
  refbox.lindatRefBox({
    rest: 'https://lindat.mff.cuni.cz/repository/rest'
  });
}

// --- Main Initialization ---
function init() {
    console.log("Initializing App...");

    // Initialize Tabs
    initTabs();

    // 1. Setup Form Handling (Vanilla JS)
    const form = document.getElementById('classifyForm');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const resultDiv = document.getElementById('results');
            const loader = document.getElementById('loading');

            // UI Reset
            resultDiv.innerHTML = '';
            loader.style.display = 'block';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errJson = await response.json();
                    throw new Error(errJson.detail || `Server Error: ${response.statusText}`);
                }

                const data = await response.json();
                renderResults(data);
            } catch (err) {
                console.error(err);
                resultDiv.innerHTML = `<div class="model-card" style="border-color: red; color: red;">
                    <strong>Error:</strong> ${err.message}
                </div>`;
            } finally {
                loader.style.display = 'none';
            }
        });
    } else {
        console.error("Form element 'classifyForm' not found!");
    }

    // 2. Setup Legacy LINDAT settings
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