# PicWise — Ingredient & Nutrition Region Detection (v1)

> **v1.2 update:** Ingredients and Nutrition are now detected as two
> **fully independent** regions. The previous version restricted
> expansion to a shared global "column" model, which silently failed on
> narrow-gutter side-by-side layouts and caused both boxes to collapse
> into nearly the same rectangle. See "Independent detection" below.

> **v1.1 update:** the region-expansion algorithm was rewritten to fix a
> bug where detected Ingredients/Nutrition boxes only captured the
> heading instead of the full section. See "How region detection works"
> below and `output/debug_region_detection.json` (produced with
> `--test-mode`) for details on what changed.

PicWise finds the **Ingredients / Composition** section and (for food
products) the **Nutrition Information** section in a photo of a packaged
product, crops those regions, cleans them up with OpenCV, and OCRs them
into structured data.

## ⚠️ Important limitation — please read first

**This is Version 1, and there is currently no real annotated dataset of
package photographs.** So this system does **not** use a trained object
detector (no YOLO, no `best.pt`). Instead it locates regions using:

```
OpenCV  +  PaddleOCR  +  keyword/anchor heuristics  +  geometry  +  knowledge-base fuzzy matching
```

This means:

- It works out of the box, with no training step.
- Accuracy depends on OCR quality and how "normal" the package layout
  is. It has **not** been benchmarked to any accuracy number — please
  evaluate it yourself on real photographs.
- It will make mistakes on unusual layouts, heavy glare, extreme blur,
  or packages where OCR can't read the heading text at all.

The planned next phase (see "Future: YOLO" below) is to collect and
annotate real package photos and train a proper detector once that data
exists, then compare it against this OCR-anchor method and likely build
a hybrid of the two.

---

## 1. Installation

```bash
cd picwise_region_detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Notes:

- **PaddleOCR / PaddlePaddle version compatibility matters.** The
  `requirements.txt` pins a combination that is stable at time of
  writing (PaddleOCR 2.7–2.8 with PaddlePaddle 2.6). If you hit an
  install error, check the current compatibility table at
  https://github.com/PaddlePaddle/PaddleOCR before forcing different
  versions.
- If you have an NVIDIA GPU and want GPU inference, install
  `paddlepaddle-gpu` (matching your CUDA version) instead of the CPU
  `paddlepaddle` package.
- `rapidfuzz` is used for fuzzy text matching. The code has a pure-Python
  fallback if it's missing, but installing it is strongly recommended
  for speed and accuracy.

## 2. Adding your knowledge-base files

Put your three files here, using **exactly these filenames** (or edit
the paths in `config.py` if you'd rather keep your own names):

```
picwise_region_detector/knowledge_base/ingredient_knowledge_base_500_with_alternate_names.csv
picwise_region_detector/knowledge_base/nutrition_knowledge_dataset.csv
picwise_region_detector/knowledge_base/personal_care_ingredients_dataset_csv.xlsx
```

You do **not** need to know or match exact column names. On first load,
`matching/knowledge_base.py` prints the discovered columns and its
best guess for which column is the "name" column and which (if any) is
the "alternate names" column, e.g.:

```
[knowledge_base] Available columns: ['ingredient_name', 'alt_names', 'category']
[knowledge_base] Guessed name_column='ingredient_name', alt_column='alt_names'
```

If the guess looks wrong for your file, open
`matching/knowledge_base.py` and edit `NAME_COLUMN_HINTS` /
`ALT_NAME_COLUMN_HINTS` to include your actual column name, or rename
the column in your CSV/XLSX.

If a knowledge-base file is missing, PicWise still runs — it just skips
fuzzy-matching against that file (ingredients are still extracted from
OCR text, they just won't have a `matched_name`).

## 3. Running

```bash
python main.py --image input/product.jpg
```

Optional flags:

```bash
python main.py --image input/product.jpg --output output/
python main.py --image input/product.jpg --domain auto            # default
python main.py --image input/product.jpg --domain food
python main.py --image input/product.jpg --domain personal_care
python main.py --image input/product.jpg --test-mode              # save every intermediate stage
```

## 4. Expected output

```
output/
    original.jpg               # your input image, copied as-is
    packet_crop.jpg             # OpenCV-isolated packaging (or full image if low confidence)
    ingredients_region.jpg      # raw crop of the detected ingredients area
    nutrition_region.jpg        # raw crop of the detected nutrition area (food domain only)
    ingredients_processed.jpg   # best-scoring preprocessing variant for ingredients
    nutrition_processed.jpg     # best-scoring preprocessing variant for nutrition
    visualization.jpg           # original image with GREEN/BLUE/RED boxes
    result.json                 # full structured result
    result.txt                  # human-readable summary + processing log
    stages/                     # only with --test-mode: every intermediate image
```

`visualization.jpg` color key:

- **GREEN** = Ingredients / Composition region
- **BLUE**  = Nutrition Information region
- **RED**   = other OCR text boxes not assigned to either region

`result.json` shape:

```json
{
  "image": "product.jpg",
  "domain": "food",
  "ingredients_region": {"bbox": [x1,y1,x2,y2], "confidence": 0.91, "anchor": "ingredients", "method": "anchor_expansion"},
  "nutrition_region": {"bbox": [x1,y1,x2,y2], "confidence": 0.88, "anchor": "nutrition information", "method": "anchor_expansion"},
  "ingredients": [{"ocr_text": "whole wheat flour", "matched_name": "Whole Wheat Flour", "confidence": 0.94}],
  "nutrition": {"energy": {"value": 450, "unit": "kcal"}, "protein": {"value": 8, "unit": "g"}},
  "processing": {"best_ingredient_variant": "clahe", "best_nutrition_variant": "adaptive_threshold", "elapsed_seconds": 3.21},
  "image_quality": {"laplacian_variance": 143.2, "brightness_mean": 128.5, "is_blurry": false, "is_too_dark": false, "is_too_bright": false}
}
```

## 5. How the region-detection algorithm works (no training data needed)

### Independent detection (v1.2)

Ingredients and Nutrition are detected by two completely independent
pipelines that never look at each other's anchor, bbox, or geometry:

- `detection/ingredient_region.py` finds the top few Ingredients-heading
  candidates anywhere in the image and expands each on its own, across
  the *entire* line list (no shared column restriction).
- `detection/nutrition_region.py` does the same for Nutrition headings
  and, as a fallback, nutrient-keyword table clustering.
- The only thing keeping them apart is **semantics, not geometry**:
  `detection/section_signals.py` classifies every candidate line as a
  strong Nutrition row (nutrient keyword + number/unit, e.g. "Protein
  8 g") or a strong Ingredients line (comma-separated list / vocabulary
  hits). Each detector's expansion loop disqualifies any line that looks
  like the *other* section's content, no matter how spatially close it
  is - this is what correctly separates a Nutrition table sitting right
  next to an Ingredients paragraph even when the gutter between them is
  narrow.
- After both are found, `detection/region_reconciliation.py` checks
  their IoU as a final safety net; only if it's still too high does it
  reassign contested lines and shrink both boxes - it never merges them
  into one box.
- If only one section is visible, the other is correctly returned as
  `bbox: null` rather than a fabricated region.

This replaces the earlier `cluster_into_columns()`-gated approach, which
assumed Ingredients/Nutrition always sat in visually separated columns
with a wide gutter - real packets often have a much narrower gutter (or
none at all in a stacked layout), which silently broke that assumption
and caused both boxes to collapse into nearly the same rectangle.

### What changed in v1.1 (fixing "only the heading gets cropped")

The original expansion walked raw, individual OCR boxes and required
each new line to overlap the *anchor's* narrow rect, which stalled after
1-2 lines on real photos. The rewrite (`detection/geometry.py`,
`detection/ingredient_region.py`, `detection/nutrition_region.py`) adds:

- **Line grouping** (`group_ocr_boxes_into_lines`): fragmented OCR boxes
  on the same visual row (e.g. a nutrient name box and its separate
  value/unit box) are merged into one line first, so expansion reasons
  about whole lines, not arbitrary fragments.
- **Column clustering** (`cluster_into_columns`): lines are split into
  left-to-right page columns wherever a wide gutter is detected, so a
  side-by-side Nutrition table and Ingredients paragraph can never bleed
  into each other during expansion.
- **Band-based expansion** instead of strict overlap: a growing
  left-aligned "column band" accepts a new line if it starts near the
  paragraph's established left margin OR overlaps the band built so far
  - this is what lets wrapping ingredient lines of very different widths
    still get included.
- **All lines in the column are walkable**, not just lines containing a
  recognized keyword - this is what lets Nutrition headers ("Serving
  Size: 30 g") and footnotes ("*Percent Daily Values...") get absorbed
  into the table even though they don't contain a nutrient keyword
  themselves.
- **Contextual stop-word validation** (`detect_section_boundaries`):
  `"contains"` only triggers a stop when the line also mentions a common
  allergen term (milk, wheat, soy, nuts, ...); other stop words need
  either a high fuzzy-match score or a short, heading-like line to be
  trusted, so a stray substring match inside a long ingredient line won't
  falsely truncate the section.
- **Multi-signal confidence scoring** (`score_ingredient_region`,
  `score_nutrition_region`): combines anchor-match quality, number of
  lines gathered, vocabulary/nutrient-keyword density, and boundary
  quality (did expansion stop on a genuine heading, or just run out of
  nearby lines?) instead of using anchor confidence alone.
- **Debug output**: pass `debug=True` (or run `main.py --test-mode`) to
  get a `debug` dict per region with the anchor used, every rejected
  line and why, the stop reason, and the final column band - written to
  `output/debug_region_detection.json`.


### Step-by-step

1. **Load & normalize** the image (upscale if OCR-too-small, downscale
   if huge, always keeping aspect ratio).
2. **Optional packet isolation**: pure OpenCV edge/contour analysis
   (`detection/packet_region.py`) tries to find the packaging boundary
   and crop to it. If confidence is low, it uses the full image instead
   of risking a bad crop — this is a fallback heuristic, not a trained
   detector.
3. **Full-image OCR** with PaddleOCR, returning text + confidence + a
   4-point bounding box per detected text line.
4. **Anchor search**: every OCR text box is fuzzy-matched (via
   `rapidfuzz`) against known heading vocabulary — "INGREDIENTS",
   "COMPOSITION", "NUTRITION INFORMATION", Hindi terms, etc. Fuzzy
   matching means OCR noise like "INGRED1ENTS" or "INGREDIENTS-" is
   still recognized.
5. **Region expansion**: starting at the best anchor match, PicWise
   walks through the remaining OCR boxes in reading order and grows a
   bounding region by checking, for each candidate line: is it below/
   right of the anchor, is the vertical/horizontal gap small enough
   (relative to the median line height), and does it horizontally/
   vertically line up with the block built so far? It keeps absorbing
   lines until either a genuine new section heading is detected (from a
   stop-word list: "nutrition", "allergen information", "storage",
   "batch no", etc.) with high confidence, or a line-count/gap budget is
   exhausted.
6. **Fallback without a heading** (`STEP 7` / `STEP 22`): if no
   "INGREDIENTS" or "NUTRITION" heading is found at all (common with
   cropped or damaged photos), PicWise instead looks for **clusters of
   known vocabulary**:
   - For nutrition: OCR boxes containing nutrient keywords (energy,
     protein, fat, sodium, ...) are clustered by proximity, and clusters
     are ranked by how many distinct nutrient terms they contain and how
     "table-like" their geometry is (consistent line heights, compact
     bounding box, short lines rather than paragraphs).
   - For ingredients: OCR boxes containing known ingredient names (from
     your knowledge-base CSV) are clustered the same way.
7. **ROI preprocessing**: each detected region is deskewed (only if
   skew-angle confidence is high enough), perspective-corrected (only if
   a strong quadrilateral distortion is found), then expanded into 8
   different OpenCV-enhanced variants (CLAHE, sharpen, OTSU, adaptive
   threshold, denoise, combinations).
8. **OCR ensemble**: PaddleOCR runs again on each of the 8 variants. The
   best one is chosen not by raw OCR confidence alone but by a combined
   score that also rewards ingredient/nutrient keyword hits, sensible
   line structure, and unit/number presence, and penalizes garbage
   characters — see `ocr/ensemble.py`.
9. **Parsing**: the winning OCR text is parsed into a clean ingredient
   list (comma/semicolon-aware, INS/E-number and percentage stripping,
   never splitting on bare whitespace so multi-word names like "Sodium
   Benzoate" stay intact) or a structured nutrition dict
   (`{"energy": {"value": 450, "unit": "kcal"}, ...}`).
10. **Knowledge-base fuzzy matching**: each parsed ingredient token is
    matched against your CSV/XLSX vocabulary (again via `rapidfuzz`),
    correcting OCR noise like "Citric Acld" → "Citric Acid" — but only
    when the match similarity clears a configurable threshold, so
    garbage tokens are left unmatched rather than forced onto a random
    KB entry.
11. **Visualization + JSON/TXT output** are written to `output/`.

### Why this works without a trained detector

Ingredient and nutrition panels are **highly standardized, text-heavy,
regulatorily-mandated layouts**. Unlike general object detection (where
you genuinely need to learn visual appearance), this task has strong
*textual* structure we can exploit directly: known heading words, known
nutrient vocabulary, and known ingredient names. OCR + fuzzy text
matching + simple geometric grouping gets you a long way precisely
*because* the target regions are defined by their text content, not
their visual appearance. This is a deliberate, pragmatic choice for v1,
not a claim that it's as robust as a trained detector — see the
limitation notice above.

## 6. Domain classification

`main.py`'s `classify_domain()` (STEP 20) counts hits from two small
keyword lists (`config.FOOD_SIGNALS` / `config.PERSONAL_CARE_SIGNALS`)
in the full-image OCR text and picks whichever has more hits, defaulting
to `food` on a tie. This is intentionally simple and fully configurable
in `config.py` — no neural classifier is used in v1. You can always
override it with `--domain food` / `--domain personal_care`.

## 7. Debugging instructions

- Run with `--test-mode` to get every intermediate image under
  `output/stages/<ingredients|nutrition>/`: the raw crop, the deskewed
  version, the perspective-corrected version, and all 8 preprocessing
  variants that were OCR'd.
- `result.txt` includes a full processing log (each pipeline stage's
  timing/decisions) at the bottom.
- `visualization.jpg` is the fastest way to see *why* a region was
  wrong — e.g. if the GREEN box is too small, the anchor was probably
  found but expansion stopped early (check for a false-positive stop
  word); if there's no GREEN box at all, no anchor and no vocabulary
  fallback matched, and you'll want to inspect the RED boxes (raw OCR
  output) to see what PaddleOCR actually read.
- If PaddleOCR fails to initialize, `ocr/paddle_engine.get_init_error()`
  holds the reason; `run_ocr()` degrades to returning an empty list
  rather than crashing, so check your console output for the install
  error near pipeline start.
- Knowledge-base column guesses are printed at load time (see section
  2 above) — check these first if `matched_name` is always `null`.

## 8. Project structure

```
picwise_region_detector/
├── main.py                  # CLI entry point / pipeline orchestration
├── config.py                # all tunable constants & vocabulary
├── requirements.txt
│
├── preprocessing/
│   ├── image_utils.py       # load/normalize/crop/quality-check
│   ├── enhancement.py       # CLAHE/sharpen/threshold/denoise variants
│   ├── deskew.py             # Hough + min-area-rect skew estimation & correction
│   └── perspective.py        # conditional 4-point perspective correction
│
├── detection/
│   ├── ocr_detector.py      # text normalization + fuzzy anchor matching + full-image OCR
│   ├── ingredient_region.py # ingredient anchor search + region expansion + fallback
│   ├── nutrition_region.py  # nutrition anchor search + table-geometry fallback
│   ├── packet_region.py     # OpenCV-only packaging isolation (no YOLO)
│   └── geometry.py          # shared spatial helpers (distance, overlap, clustering)
│
├── ocr/
│   ├── paddle_engine.py     # PaddleOCR adapter (handles version differences)
│   └── ensemble.py          # multi-variant OCR + combined scoring
│
├── parsing/
│   ├── ingredient_parser.py # raw text -> clean ingredient list
│   └── nutrition_parser.py  # raw text -> structured nutrition dict
│
├── matching/
│   └── knowledge_base.py    # CSV/XLSX loading, column discovery, fuzzy KB matching
│
├── visualization/
│   └── draw_regions.py      # GREEN/BLUE/RED bounding-box visualization
│
├── models/
│   └── yolo_region_detector.py  # OPTIONAL, unused scaffold for future YOLO model
│
├── knowledge_base/           # <- put your 3 CSV/XLSX files here
├── input/                     # <- put product photos here
└── output/                    # <- results are written here
```

## 9. Future: adding YOLO once you have real annotated data

```
Collect 500–1000 real Indian package images
        ↓
Annotate Ingredients (class 0) + Nutrition (class 1) boxes
        ↓
Train YOLO (e.g. with the `ultralytics` package)
        ↓
Point models/yolo_region_detector.py's YoloRegionDetector at the trained weights
        ↓
Compare YOLO vs OCR-anchor method on a held-out test set
        ↓
Build a hybrid detector (e.g. YOLO for the initial ROI proposal,
OCR-anchor heuristics as a confidence check / refinement / fallback)
```

`models/yolo_region_detector.py` is a fully-scaffolded but **inactive**
module: it is never imported by `main.py`, and instantiating
`YoloRegionDetector` without real weights raises a clear
`FileNotFoundError` rather than pretending to work. This keeps v1 honest
about what it actually does.

## 10. Test mode

`--test-mode` saves every intermediate processing stage (see section 7)
so you can visually audit exactly what the pipeline did to each region
before OCR, which preprocessing variant OCR'd best, and why.
