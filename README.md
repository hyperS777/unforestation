# TerraGuard: Western Ghats Forest & Deforestation Intelligence System

An advanced Earth Observation (EO) and Deep Learning system designed to detect tree-cover dynamics, forest degradation, and deforestation across the Western Ghats (Pune, Satara, and Sindhudurg) using Sentinel-2 10m multi-spectral imagery.

---

## 🚀 Key Improvements & Accuracy Upgrades

### 1. Model Architecture & Accuracy Upgrades
- **Attention Res-UNet with ASPP (Atrous Spatial Pyramid Pooling):**
  - **Atrous Spatial Pyramid Pooling (ASPP):** Multi-scale dilated convolutions at rates `[1, 6, 12, 18]` at the bottleneck capture features across multiple scales (micro-clearings, narrow logging roads, and continuous forest reserves) without losing spatial resolution.
  - **Residual Shortcut Connections:** In each convolution block to eliminate gradient degradation and preserve fine canopy texture details.
  - **Additive Attention Gates (AGs):** Positioned on decoder skip connections to focus attention on tree canopies while suppressing background agricultural and bare soil noise.
  - **Spatial Dropout:** Enforces spatial feature regularization.
- **Focal Tversky & Hybrid Compound Loss:**
  - Implemented **Focal Tversky Loss** ($\beta=0.7, \alpha=0.3, \gamma=0.75$) specifically penalizing False Negatives to prevent missing narrow logging incisions.
  - Compound **Dice-BCE Loss** combining Soft Dice Loss (optimizing direct IoU overlap) and Focal BCE (sharpening hard boundary pixels).
- **8-Fold Data Augmentation (D4 Symmetry Group):** Expanded the initial 32 patches into 256 high-variance patches with 90°/180°/270° rotations, horizontal/vertical flips, and contrast perturbations.
- **Seamless Tile Inference (Hann Windowing + TTA):** Integrated 2D Hann cosine bell window blending to eliminate rectangular tile boundary seams, plus 4-fold Test-Time Augmentation (TTA) for higher prediction stability.
- **Adaptive Validation Threshold Tuning:** Replaced the arbitrary 0.5 threshold with an automated validation search identifying the optimal threshold $\tau^*$ maximizing F1 / Dice score.

### 2. Streamlit Dashboard (`app.py`) Overhaul
- **Dark Mission-Control Theme:** High-contrast glassmorphism layout with custom CSS, responsive metric cards, and alert badges.
- **Two Operating Modes:**
  - **Demo Mode:** Explore built-in Western Ghats datasets (Pune, Satara, Sindhudurg) immediately without needing to upload files.
  - **Custom GeoTIFF Upload:** Upload Colab output GeoTIFFs with automated tag detection.
- **↔️ Interactive Draggable Split-Screen Slider (Swipe Inspector):**
  - Draggable vertical divider to compare 2021 vs 2024 canopy change masks in real time.
- **🗺️ Interactive Geospatial Map (Folium / Leaflet):**
  - Satellite basemap (Esri World Imagery) with district AOI boundaries.
  - Multi-layer overlays (2021 Forest, 2024 Forest, Categorical Change Map, NDVI Delta).
  - Deforestation Hotspot Pins with interactive popups showing cluster area, pixel count, Carbon impact, and GPS coordinates.
- **🔥 Carbon Stock & Climate Impact Modeling:**
  - Estimates gross carbon emissions ($tCO_2e$), annual sequestration by regrowth, and mature tree equivalents based on Western Ghats biomass densities (~145 $tC/ha$).
- **MMU Noise Filtering Slider:** Tune the Minimum Mapping Unit (1 to 12 pixels) to eliminate sensor speckle while preserving true canopy tracks.
- **📊 Quantitative Analytics (Plotly):**
  - 2021-2024 Landcover partition donut chart (Stable Forest, Deforestation Loss, Regrowth, Non-Tree).
  - Canopy turnover bar chart and cross-district benchmark.
- **🚨 Deforestation Hotspot Catalog:**
  - Filterable cluster table (Critical, High, Medium, Low severity) with one-click CSV export.
- **📑 Executive Report Hub:**
  - Auto-generates Markdown audit summaries and downloadable KPI datasets.

---

## 📂 Project Structure

```
ml/
├── app.py                          # Modern TerraGuard Streamlit Dashboard
├── model.py                        # Attention Res-UNet + ASPP architecture & hybrid losses
├── augmentation.py                 # D4 symmetry & perturbation data augmentation
├── inference.py                    # Smooth 2D Hann window blending & 4x TTA engine
├── analytics.py                    # Geodesic pixel areas (ha), change matrix, MMU filter, & Carbon modeling
├── sample_data.py                  # High-fidelity synthetic dataset generator & demo engine
├── sample_data/                    # Pre-generated sample GeoTIFFs (18 files)
├── Deforestation_project.ipynb     # Upgraded End-to-End Colab Notebook
├── Deforestation_project.backup.ipynb # Original notebook backup
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 💻 Running the Dashboard

```bash
streamlit run ml/app.py
```

Then open `http://localhost:8501` in your browser.
