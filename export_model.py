import joblib, numpy as np, json, struct, warnings
warnings.filterwarnings("ignore")

model = joblib.load('/mnt/user-data/uploads/random_forest_risk_model.pkl')
names = list(model.feature_names_in_)
n_features = len(names)

feature_arr = []
threshold_arr = []
left_arr = []
right_arr = []
proba_arr = []  # probability of class 1 at leaf, 0 elsewhere
tree_offsets = [0]

for est in model.estimators_:
    t = est.tree_
    offset = len(feature_arr)
    n = t.node_count
    feat = t.feature  # -2 for leaf
    thresh = t.threshold
    cl = t.children_left
    cr = t.children_right
    val = t.value  # shape (n_nodes, 1, n_classes)
    for i in range(n):
        feature_arr.append(int(feat[i]))
        threshold_arr.append(float(thresh[i]))
        left_arr.append(int(cl[i]) + offset if cl[i] != -1 else -1)
        right_arr.append(int(cr[i]) + offset if cr[i] != -1 else -1)
        if feat[i] == -2:  # leaf
            counts = val[i][0]
            p1 = counts[1] / counts.sum() if counts.sum() > 0 else 0.0
        else:
            p1 = 0.0
        proba_arr.append(float(p1))
    tree_offsets.append(offset + n)

n_nodes = len(feature_arr)
print("total nodes:", n_nodes, "trees:", len(tree_offsets)-1)

# Pack into binary buffers
feature_bytes = struct.pack(f'<{n_nodes}h', *feature_arr)          # int16
threshold_bytes = struct.pack(f'<{n_nodes}f', *threshold_arr)      # float32
left_bytes = struct.pack(f'<{n_nodes}i', *left_arr)                # int32
right_bytes = struct.pack(f'<{n_nodes}i', *right_arr)              # int32
proba_bytes = struct.pack(f'<{n_nodes}f', *proba_arr)              # float32
offsets_bytes = struct.pack(f'<{len(tree_offsets)}i', *tree_offsets)

import os
os.makedirs('/home/claude/site/data', exist_ok=True)
with open('/home/claude/site/data/feature.bin','wb') as f: f.write(feature_bytes)
with open('/home/claude/site/data/threshold.bin','wb') as f: f.write(threshold_bytes)
with open('/home/claude/site/data/left.bin','wb') as f: f.write(left_bytes)
with open('/home/claude/site/data/right.bin','wb') as f: f.write(right_bytes)
with open('/home/claude/site/data/proba.bin','wb') as f: f.write(proba_bytes)
with open('/home/claude/site/data/offsets.bin','wb') as f: f.write(offsets_bytes)

meta = {
    "n_nodes": n_nodes,
    "n_trees": len(tree_offsets) - 1,
    "n_features": n_features,
    "feature_names": names,
}
with open('/home/claude/site/data/meta.json','w') as f:
    json.dump(meta, f)

print("done")
