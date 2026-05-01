from pathlib import Path

PAIRS = [
    # Financials
    ("JPM",  "BAC"),   # big banks
    ("WFC",  "C"),     # banks
    ("GS",   "MS"),    # investment banks
    ("V",    "MA"),    # payment networks
    ("BLK",  "STT"),   # passive asset managers
    ("MET",  "PRU"),   # life insurance

    # Energy
    ("XOM",  "CVX"),   # oil majors
    ("COP",  "CVX"),   # E&P vs integrated — same crude driver
    ("VLO",  "MPC"),   # oil refiners — crack spread proxy
    ("SLB",  "HAL"),   # oilfield services
    ("NEE",  "DUK"),   # electric utilities
    ("SO",   "DUK"),   # regulated utilities southeast

    # Consumer Staples
    ("KO",   "PEP"),   # beverages
    ("PG",   "CL"),    # household & personal care

    # Consumer Discretionary
    ("HD",   "LOW"),   # home improvement
    ("MCD",  "QSR"),   # fast food
    ("RCL",  "CCL"),   # cruise lines
    ("TGT",  "WMT"),   # mass retail

    # Healthcare
    ("UNH",  "CI"),    # health insurers — managed care duopoly
    ("MRK",  "PFE"),   # big pharma
    ("ABT",  "MDT"),   # medical devices
    ("CVS",  "WBA"),   # pharmacy retail

    # Industrials
    ("DAL",  "UAL"),   # airlines
    ("UPS",  "FDX"),   # logistics
    ("LMT",  "RTX"),   # defense prime
    ("GD",   "NOC"),   # defense prime
    ("CSX",  "NSC"),   # eastern railroads
    ("CAT",  "DE"),    # heavy machinery
    ("T",    "VZ"),    # telecom
    ("F",    "GM"),    # US automakers

    # Materials & Mining
    ("NEM",  "AEM"),   # gold miners
    ("FCX",  "SCCO"),  # copper miners

    # Semiconductors (Equipment)
    ("AMAT", "LRCX"),  # wafer fab equipment

    # Media & Streaming
    ("DIS",  "CMCSA"), # Disney vs Comcast — content + distribution duopoly
]

TICKERS = list(dict.fromkeys(t for pair in PAIRS for t in pair))

START_DATE = "2024-04-01"
END_DATE = "2026-04-28"

REQUEST_SLEEP = 25

RTH_START = "09:30"
RTH_END = "16:00"
SIGNAL_START = "10:00"

RECENT_BARS = 2400

CORR_THRESHOLD = 0.5
CORR_TOP_N = 10
COINT_TOP_N = 3

ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 3.5
ENTRY_Z_VOLATILE = 2.8   # stricter threshold when HMM detects volatile regime

# Transaction costs (per side, per leg, as fraction of price)
COST_COMMISSION = 0.0003   # broker commission (e.g. IBKR tiered)
COST_SPREAD     = 0.0003   # half of bid-ask spread (liquid large-caps)
COST_SLIPPAGE   = 0.0001   # slippage: signal on close, fill near open
COST_PER_SIDE   = COST_COMMISSION + COST_SPREAD + COST_SLIPPAGE  # 0.07% per side

BORROW_RATE_ANNUAL = 0.015  # 1.5% annual short borrow (liquid stocks)

PAIR_MAX_LOSS = -30.0      # disable pair if cumulative net P&L drops below this

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
