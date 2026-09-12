# Slope Risk — Landslide Movement-Type Risk Predictor

A static, client-side web app that runs your trained `random_forest_risk_model.pkl`
(300-tree RandomForestClassifier) entirely in the browser — no server, no backend,
just HTML/CSS/JS. Safe to host on GitHub Pages as-is.

## Files

| File | Purpose |
|---|---|
| `index.html` | Page structure and form |
| `style.css` | Styling |
| `script.js` | Builds the feature vector from the form and runs the forest |
| `model_data.js` | The 300 trees exported from the `.pkl`, as flat arrays (~5 MB) |
| `categories.js` | Dropdown option lists, derived from the model's own trained categories |
| `export_model.py` | The Python script that generated `model_data.js` / `categories.js` — re-run this any time you retrain the model |

## How it works

GitHub Pages only serves static files — it cannot run Python or load a `.pkl`
directly. So `export_model.py` walks the trained `RandomForestClassifier` and
writes out each tree's structure (feature index, split threshold, left/right
child pointers, and leaf probabilities) as plain JavaScript arrays. `script.js`
then re-implements the same traversal sklearn uses internally: walk each tree
to a leaf, average the "probability of High risk" across all 300 trees, and
classify High if that average is ≥ 0.5. This was verified to match
`model.predict_proba()` output exactly (to 6+ decimal places) on test inputs.

**Label:** High risk = Movement Type is Slide or Fall. Low risk = anything else
(Flow, Subsidence, Topple, Composite, etc.) — matching how the model was trained.

**Dropdown options** are pulled directly from the columns the model was
trained on, so a selection always corresponds to something the model actually
saw. A few corrupted category values that were present in the source data
(long garbled text fragments, stray digits) were filtered out of the dropdowns
in `export_model.py` — they still exist internally in the model but aren't
real states/districts, so they're not offered as choices.

## Deploying to GitHub Pages

1. Create a new GitHub repository (or use an existing one).
2. Add all the files in this folder to the repo root (or to a `/docs` folder —
   your choice, just set the Pages source accordingly).
3. Commit and push:
   ```bash
   git init
   git add index.html style.css script.js model_data.js categories.js
   git commit -m "Add landslide risk predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
4. In the repo on GitHub: **Settings → Pages → Source**, choose the `main`
   branch and `/ (root)` folder, then save.
5. GitHub gives you a URL like `https://<your-username>.github.io/<your-repo>/`
   within a minute or two.

## Retraining later

If you retrain the model (new data, different features, different
hyperparameters), just re-run:

```bash
python3 export_model.py
```

with `random_forest_risk_model.pkl` in the same folder — it regenerates
`model_data.js` and `categories.js` to match. `index.html`/`script.js` don't
need to change unless you add or rename features.

## Note on file size

`model_data.js` is ~5 MB because the forest has 300 fully-grown trees
(~226,000 nodes total). That's fine for GitHub Pages (no practical size limit
for a file this size) but it does mean the page has a one-time ~5 MB download
on first load. If you want something lighter, consider retraining with
`max_depth` capped (e.g. 10–15) — smaller trees, smaller file, similar
accuracy in most cases.
