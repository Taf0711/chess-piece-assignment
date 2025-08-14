# GENERATED USING GOOD OL GPT SOOOUP

import json
import re

# === Config ===
coco_path = "./coco_data/coco_annotations.json"

# === Shorthand mapping ===
SHORTHAND = {
    "WhitePawn": "WP",
    "WhiteRook": "WR",
    "WhiteKnight": "WN",
    "WhiteBishop": "WB",
    "WhiteQueen": "WQ",
    "WhiteKing": "WK",
    "BlackPawn": "BP",
    "BlackRook": "BR",
    "BlackKnight": "BN",
    "BlackBishop": "BB",
    "BlackQueen": "BQ",
    "BlackKing": "BK",
    "Board": "Board",
    "Cylinder": "Cylinder"
}

# === Load the COCO JSON ===
with open(coco_path, "r") as f:
    coco = json.load(f)

# === Step 1: Build old_name → new_shorthand_name mapping ===
old_to_short = {}
for cat in coco["categories"]:
    base_name = re.sub(r'\d+$', '', cat["name"])  # "WhitePawn8" → "WhitePawn"
    short = SHORTHAND.get(base_name)
    if short is None:
        print(f"️  Unrecognized category base name: {base_name}")
        short = base_name  # fallback
    old_to_short[cat["name"]] = short

# === Step 2: Build new name → new id, and old id → new id mapping ===
new_name_to_id = {}
old_id_to_new_id = {}
new_categories = []
next_id = 1

for cat in coco["categories"]:
    old_id = cat["id"]
    short_name = old_to_short[cat["name"]]

    if short_name not in new_name_to_id:
        new_name_to_id[short_name] = next_id
        new_categories.append({
            "id": next_id,
            "name": short_name,
            "supercategory": "chess"
        })
        next_id += 1

    old_id_to_new_id[old_id] = new_name_to_id[short_name]

# === Step 3: Update annotation category_ids ===
for ann in coco["annotations"]:
    ann["category_id"] = old_id_to_new_id[ann["category_id"]]

# === Step 4: Replace the category list ===
coco["categories"] = new_categories

# === Save updated JSON ===
with open(coco_path, "w") as f:
    json.dump(coco, f, indent=2)

print(" COCO categories renamed using shorthand.")
