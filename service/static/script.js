/**
 * Renders the prediction results into the DOM.
 * Expects data structure: { predictions: { "v1.3": { label: "Typed", score: 0.99 }, ... } }
 */
function renderResults(data) {
    const container = document.getElementById('results');
    container.innerHTML = ''; // Clear previous results (safety check)

    if (!data || !data.predictions) {
        container.innerHTML = `<div class="alert alert-warning">No predictions returned from server.</div>`;
        return;
    }

    const predictions = data.predictions;
    
    Object.keys(predictions).forEach(version => {
        const pred = predictions[version];
        
        // Handle potential error in single model
        if(pred.error) {
            container.innerHTML += `<div class="model-card"><strong>${version}</strong>: Error - ${pred.error}</div>`;
            return;
        }

        let html = `<div class="model-card"><h3>${version}</h3>`;
        
        // Handle both single object or array (top-N)
        const items = Array.isArray(pred) ? pred : [pred];
        
        items.forEach(item => {
            const scorePct = (item.score * 100).toFixed(2);
            // Dynamic color for score bar (Green > 90%, Yellow > 50%, Red < 50%)
            const color = item.score > 0.9 ? '#4CAF50' : (item.score > 0.5 ? '#FFC107' : '#F44336');
            
            html += `
                <div style="margin-bottom: 0.5rem;">
                    <div style="display:flex; justify-content:space-between;">
                        <span>${item.label}</span>
                        <span>${scorePct}%</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${scorePct}%; background-color: ${color};"></div>
                    </div>
                </div>
            `;
        });
        html += `</div>`;
        container.innerHTML += html;
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

    // 1. Setup Form Handling (Vanilla JS) - PRIORITY
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
                // Determine if we need to call a specific endpoint based on selection
                // (Currently assumes a single /predict endpoint handles logic)
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error(`Server Error: ${response.statusText}`);
                
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

    // Safely attempt legacy injects
    try {
        injectStylesheets(url);
        switchHandle(handle, title);
    } catch(e) {
        console.warn("LINDAT stylesheet injection failed:", e);
    }

    document.body.setAttribute('id', project);

    // 3. AngularJS Legacy Handling (Safeguarded)
    // Only run this if Angular is actually loaded in the HTML
    if (typeof angular !== 'undefined') {
        var app = angular.module('lindatApp', ['lindat']);
        app.config(['$httpProvider', function($httpProvider) {
            $httpProvider.defaults.useXDomain = true;
        }]);
    } else {
        console.log("AngularJS not detected. Skipping Angular module init (this is expected for Vanilla JS app).");
    }
}

// Execute on page load using jQuery
$(init);
