# Slope Watch — Landslide Risk Predictor

A static, client-side landslide risk assessment tool. Your original
`random_forest_risk_model.pkl` (300-tree scikit-learn RandomForestClassifier)
has been exported into flat binary arrays that a small JavaScript runtime
walks directly in the browser — no Python, no server required, so it works
on GitHub Pages as-is.

## Deploying to GitHub Pages

1. Create a new GitHub repo (or use an existing one).
2. Copy everything in this folder into the repo root:
   - `index.html`
   - `style.css`
   - `app.js`
   - `data/` (the exported model — feature.bin, threshold.bin, left.bin,
     right.bin, proba.bin, offsets.bin, meta.json, config.json)
3. Commit and push.
4. In the repo settings → **Pages**, set source to "Deploy from branch",
   branch `main`, folder `/ (root)`. Save.
5. GitHub gives you a URL like `https://<username>.github.io/<repo>/` —
   that's your live site, usually within a minute or two.

No build step, no dependencies to install — it's plain HTML/CSS/JS plus
the `data/` folder (~4 MB of exported tree weights).

## How the conversion works

`export_model.py` (included) walks every tree in the forest and flattens
all ~226,000 decision nodes into six typed-array binary files: which
feature each node splits on, its threshold, the left/right child indices,
and the leaf-node probability of the positive class. `app.js` fetches
those files, reconstructs the arrays, and re-implements the same
threshold-comparison logic scikit-learn uses internally, averaged across
all 300 trees.

This was verified against the original model directly — predictions match
scikit-learn's `predict_proba` to 8 decimal places (the only difference is
float32 vs float64 storage).

## Known data quality issue

A few categories in the original training data's `Soil Distribution` and
`Vegetation` columns contain corrupted values — raw spreadsheet-row text
that leaked into the category field instead of a clean soil/vegetation
description. Those specific bad values are excluded from the dropdown
menus in the UI (so users can't accidentally select them), but they still
exist as unused columns in the model. If you retrain the model later, it's
worth cleaning those source columns first.

## Limitations to disclose to users

- This is a statistical model trained on historical slide records, not a
  geotechnical survey or real-time monitoring system.
- Latitude/longitude dominate the model's decisions — it's likely picking
  up regional/local patterns rather than a generalizable physical model,
  so predictions for locations far from the training data (outside
  Assam, Manipur, Meghalaya, Mizoram, Nagaland, Tripura) won't be meaningful.
