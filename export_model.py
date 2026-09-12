import joblib
import json
import numpy as np

model = joblib.load('/mnt/user-data/uploads/random_forest_risk_model.pkl')
feature_names = list(model.feature_names_in_)

# ---- 1. Export tree structure into flat arrays (compact) ----
trees = []
for est in model.estimators_:
    t = est.tree_
    n = t.node_count
    feature = t.feature.tolist()              # -2 = leaf
    threshold = [round(float(x), 6) for x in t.threshold]
    left = t.children_left.tolist()
    right = t.children_right.tolist()
    # probability of class 0 ("High") at each node's value array
    value = t.value  # shape (n_nodes, 1, n_classes)
    prob_high = []
    for i in range(n):
        v = value[i][0]
        total = v.sum()
        prob_high.append(round(float(v[0] / total), 6) if total > 0 else 0.0)
    trees.append({"f": feature, "t": threshold, "l": left, "r": right, "p": prob_high})

with open('site/model_data.js', 'w') as fh:
    fh.write("// Auto-exported RandomForestClassifier (300 trees) -- flat array format\n")
    fh.write("const FOREST = " + json.dumps(trees, separators=(',', ':')) + ";\n")
    fh.write("const FEATURE_NAMES = " + json.dumps(feature_names) + ";\n")

# ---- 2. Export clean categorical options for dropdowns ----
def group(prefix):
    vals = sorted(f[len(prefix):] for f in feature_names if f.startswith(prefix))
    clean = []
    for v in vals:
        vs = v.strip()
        if vs.lower() == 'unknown':
            clean.append(v)
        elif vs.isdigit():
            continue  # stray numeric fragment (corruption), not a real category, e.g. "2018"
        elif len(vs) > 100:
            continue  # leaked multi-record text blob (corruption)
        elif vs.upper() == 'RD':
            continue  # stray fragment (corruption), not a real category
        else:
            clean.append(v)  # real category -- length alone is not a corruption signal
                              # (e.g. "Mon" and "Phek" are genuine, short district names)
    return clean

categories = {
    "State": group("State_"),
    "District": group("District_"),
    "Rainfall Pattern": group("Rainfall Pattern_"),
    "Soil Distribution": group("Soil Distribution_"),
    "Vegetation": group("Vegetation_"),
}

with open('site/categories.js', 'w') as fh:
    fh.write("// Dropdown options derived from the model's trained one-hot columns\n")
    fh.write("const CATEGORIES = " + json.dumps(categories, indent=2) + ";\n")

import os
print("model_data.js size:", os.path.getsize('site/model_data.js')/1024/1024, "MB")
print("trees exported:", len(trees))
print("total nodes:", sum(len(t['f']) for t in trees))
