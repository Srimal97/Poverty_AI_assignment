"""Project-wide constants for the HIES 2019 poverty-targeting model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"

KEYS = ["psu", "snumber", "hhno"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Department of Census and Statistics (DCS) Official Poverty Line, 2019 base,
# national, Rs per person per month. Verify against the DCS poverty publication
# before final submission (see report, Section 2.3).
POVERTY_LINE = 6966

DISTRICTS = {
    11: "Colombo", 12: "Gampaha", 13: "Kalutara", 21: "Kandy", 22: "Matale",
    23: "Nuwara Eliya", 31: "Galle", 32: "Matara", 33: "Hambantota", 41: "Jaffna",
    42: "Mannar", 43: "Vavuniya", 44: "Mullaitivu", 45: "Kilinochchi",
    51: "Batticaloa", 52: "Ampara", 53: "Trincomalee", 61: "Kurunegala",
    62: "Puttalam", 71: "Anuradhapura", 72: "Polonnaruwa", 81: "Badulla",
    82: "Monaragala", 91: "Ratnapura", 92: "Kegalle",
}
PROVINCES = {1: "Western", 2: "Central", 3: "Southern", 4: "Northern", 5: "Eastern",
             6: "North Western", 7: "North Central", 8: "Uva", 9: "Sabaragamuwa"}
SECTORS = {1: "Urban", 2: "Rural", 3: "Estate"}
ETHNICITY = {1: "Sinhala", 2: "Sri Lankan Tamil", 3: "Indian Tamil", 4: "Moor/Muslim",
             5: "Burgher", 6: "Malay", 9: "Other"}
