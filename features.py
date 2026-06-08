"""
Shared schema + feature engineering for the Forest Cover Type project.

Both the training pipeline (run_all.py) and the demo (app.py) import from
here so that the exact same features are produced at train and inference time.
"""
import numpy as np
import pandas as pd

# --- The fixed 54-column schema of the UCI / scikit-learn Covertype data ---
CONTINUOUS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS = [f"Wilderness_Area{i}" for i in range(1, 5)]   # 4 binary columns
SOIL = [f"Soil_Type{i}" for i in range(1, 41)]              # 40 binary columns
RAW_COLUMNS = CONTINUOUS + WILDERNESS + SOIL               # 54 columns, fixed order

# Human-readable names for the 7 target classes
COVER_TYPES = {
    1: "Spruce / Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood / Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

# Physically reasonable ranges (used for the demo sliders)
CONTINUOUS_RANGES = {
    "Elevation": (1850, 3860, 2960),
    "Aspect": (0, 360, 155),
    "Slope": (0, 66, 14),
    "Horizontal_Distance_To_Hydrology": (0, 1400, 270),
    "Vertical_Distance_To_Hydrology": (-180, 600, 46),
    "Horizontal_Distance_To_Roadways": (0, 7120, 2350),
    "Hillshade_9am": (0, 254, 212),
    "Hillshade_Noon": (0, 254, 223),
    "Hillshade_3pm": (0, 254, 142),
    "Horizontal_Distance_To_Fire_Points": (0, 7180, 1980),
}

# The columns added by add_engineered_features() (used for the ablation study)
ENGINEERED = [
    "Euclidean_Distance_To_Hydrology",
    "Mean_Distance_To_Amenities",
    "Hillshade_Mean",
    "Hillshade_Amplitude",
    "Elevation_x_Slope",
    "Above_Water",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add six physically-motivated features to a frame holding RAW_COLUMNS."""
    out = df.copy()
    hyd_h = df["Horizontal_Distance_To_Hydrology"].astype(float)
    hyd_v = df["Vertical_Distance_To_Hydrology"].astype(float)
    shades = df[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].astype(float)

    # Straight-line distance to surface water (Pythagoras on the two legs).
    out["Euclidean_Distance_To_Hydrology"] = np.sqrt(hyd_h ** 2 + hyd_v ** 2)
    # One aggregate "human accessibility" signal.
    out["Mean_Distance_To_Amenities"] = (
        hyd_h
        + df["Horizontal_Distance_To_Roadways"].astype(float)
        + df["Horizontal_Distance_To_Fire_Points"].astype(float)
    ) / 3.0
    # Daily light-exposure summaries (illumination is a strong vegetation cue).
    out["Hillshade_Mean"] = shades.mean(axis=1)
    out["Hillshade_Amplitude"] = shades.max(axis=1) - shades.min(axis=1)
    # Elevation x steepness interaction.
    out["Elevation_x_Slope"] = df["Elevation"].astype(float) * df["Slope"].astype(float)
    # Sits above or below the water table proxy.
    out["Above_Water"] = (hyd_v >= 0).astype(int)
    return out
