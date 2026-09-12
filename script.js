// ---------------------------------------------------------------
// Populate dropdowns from the model's own trained categories
// ---------------------------------------------------------------
function fillSelect(selectEl, options, { includeOther = true } = {}) {
  selectEl.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select...";
  placeholder.disabled = true;
  placeholder.selected = true;
  selectEl.appendChild(placeholder);

  options.forEach((opt) => {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt;
    selectEl.appendChild(el);
  });

  if (includeOther) {
    const other = document.createElement("option");
    other.value = "";
    other.textContent = "Other / not listed";
    selectEl.appendChild(other);
  }
}

fillSelect(document.getElementById("state"), CATEGORIES["State"]);
fillSelect(document.getElementById("district"), CATEGORIES["District"]);
fillSelect(document.getElementById("rainfall-pattern"), CATEGORIES["Rainfall Pattern"]);
fillSelect(document.getElementById("soil"), CATEGORIES["Soil Distribution"]);
fillSelect(document.getElementById("vegetation"), CATEGORIES["Vegetation"]);

// ---------------------------------------------------------------
// Build the exact feature vector the model was trained on
// ---------------------------------------------------------------
function buildFeatureVector(values) {
  const x = new Array(FEATURE_NAMES.length).fill(0);
  const index = {};
  FEATURE_NAMES.forEach((name, i) => (index[name] = i));

  const setNumeric = (name, val) => {
    if (name in index) x[index[name]] = Number.isFinite(val) ? val : 0;
  };
  const setDummy = (prefix, category) => {
    if (!category) return; // "Other / not listed" -> leave all dummies at 0 (baseline)
    const col = prefix + category;
    if (col in index) x[index[col]] = 1;
  };

  setNumeric("Latitude", values.latitude);
  setNumeric("Longitude", values.longitude);
  setNumeric("Movement History", values.movementHistory);
  setNumeric("Annual Rainfall Normal (in mm)", values.annualRainfall);

  setDummy("State_", values.state);
  setDummy("District_", values.district);
  setDummy("Rainfall Pattern_", values.rainfallPattern);
  setDummy("Soil Distribution_", values.soil);
  setDummy("Vegetation_", values.vegetation);

  return x;
}

// ---------------------------------------------------------------
// Evaluate the exported forest (300 trees) entirely client-side
// ---------------------------------------------------------------
function predictTree(x, tree) {
  const { f, t, l, r, p } = tree;
  let node = 0;
  while (f[node] !== -2) {
    node = x[f[node]] <= t[node] ? l[node] : r[node];
  }
  return p[node]; // probability of "High" (class 0) at this leaf
}

function predictForest(x) {
  let sum = 0;
  for (let i = 0; i < FOREST.length; i++) {
    sum += predictTree(x, FOREST[i]);
  }
  return sum / FOREST.length; // averaged probability of High risk
}

// ---------------------------------------------------------------
// Wire up the form
// ---------------------------------------------------------------
const form = document.getElementById("risk-form");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const verdictEl = document.getElementById("verdict");
const verdictLabel = document.getElementById("verdict-label");
const verdictNote = document.getElementById("verdict-note");
const probFillHigh = document.getElementById("prob-fill-high");
const probHighPct = document.getElementById("prob-high-pct");
const probLowPct = document.getElementById("prob-low-pct");

form.addEventListener("submit", (e) => {
  e.preventDefault();

  const values = {
    state: document.getElementById("state").value,
    district: document.getElementById("district").value,
    latitude: parseFloat(document.getElementById("latitude").value),
    longitude: parseFloat(document.getElementById("longitude").value),
    rainfallPattern: document.getElementById("rainfall-pattern").value,
    soil: document.getElementById("soil").value,
    vegetation: document.getElementById("vegetation").value,
    annualRainfall: parseFloat(document.getElementById("annual-rainfall").value),
    movementHistory: parseFloat(document.getElementById("movement-history").value),
  };

  const x = buildFeatureVector(values);
  const probHigh = predictForest(x);
  const probLow = 1 - probHigh;
  const isHigh = probHigh >= 0.5;

  resultEmpty.hidden = true;
  resultContent.hidden = false;

  verdictEl.className = "verdict " + (isHigh ? "is-high" : "is-low");
  verdictLabel.textContent = isHigh ? "High risk (Slide / Fall)" : "Low risk (other movement type)";

  probFillHigh.style.width = (probHigh * 100).toFixed(1) + "%";
  probHighPct.textContent = (probHigh * 100).toFixed(1) + "%";
  probLowPct.textContent = (probLow * 100).toFixed(1) + "%";

  verdictNote.textContent = isHigh
    ? "Historical records for similar conditions were dominated by slide or fall events, which tend to be fast-moving and give less warning."
    : "Historical records for similar conditions were more often flows, subsidence, topples, or complex movements rather than slides or falls.";
});
