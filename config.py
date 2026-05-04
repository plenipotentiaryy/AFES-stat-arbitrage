from pathlib import Path

PAIRS = [
    # ═══════════════════════════════════════════════════════════════════
    # FINANCIALS  (18 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("JPM",  "BAC"),    # big banks — top 2 by assets
    ("WFC",  "C"),      # banks — #3 vs #4
    ("JPM",  "WFC"),    # big bank cross
    ("GS",   "MS"),     # investment banks
    ("V",    "MA"),     # payment networks — Visa vs Mastercard
    ("BLK",  "STT"),    # passive asset managers
    ("MET",  "PRU"),    # life insurance
    ("USB",  "PNC"),    # super-regional banks
    ("SCHW", "MS"),     # brokerage / wealth management
    ("AIG",  "MET"),    # diversified insurance
    ("TFC",  "USB"),    # regional banks
    ("ICE",  "CME"),    # exchanges — futures & derivatives
    ("AXP",  "COF"),    # consumer finance [Top100]
    ("AXP",  "SYF"),    # consumer finance [Top100]
    ("COF",  "SYF"),    # consumer finance [Top100]
    ("AIG",  "AIZ"),    # multi-line insurance [Top100]
    ("AIG",  "L"),      # multi-line insurance [Top100]
    ("AIZ",  "L"),      # multi-line insurance [Top100]
    ("SCHW", "RJF"),    # investment banking [Top100]
    ("GS",   "RJF"),    # investment banking [Top100]
    ("MS",   "RJF"),    # investment banking [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # ENERGY  (15 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("XOM",  "CVX"),    # oil majors
    ("COP",  "CVX"),    # E&P vs integrated
    ("VLO",  "MPC"),    # oil refiners — crack spread proxy
    ("SLB",  "HAL"),    # oilfield services
    ("OXY",  "DVN"),    # E&P — Permian Basin
    ("EOG",  "COP"),    # E&P — light oil
    ("PSX",  "VLO"),    # refiners cross
    ("NEE",  "DUK"),    # electric utilities
    ("SO",   "DUK"),    # regulated utilities southeast
    ("AES",  "NEE"),    # utilities — renewables focus
    ("MPC",  "PSX"),    # oil refining & marketing [Top100]
    ("BKR",  "SLB"),    # oilfield services [Top100]
    ("BKR",  "HAL"),    # oilfield services [Top100]
    ("KMI",  "OKE"),    # oil & gas pipelines [Top100]
    ("KMI",  "WMB"),    # oil & gas pipelines [Top100]
    ("OKE",  "WMB"),    # oil & gas pipelines [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # TECHNOLOGY  (14 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("MSFT", "AAPL"),   # mega-cap tech — OS/ecosystem duopoly
    ("GOOGL","META"),   # digital advertising duopoly
    ("CRM",  "NOW"),    # enterprise SaaS — Salesforce vs ServiceNow
    ("ADBE", "CRM"),    # enterprise cloud software
    ("ORCL", "IBM"),    # legacy enterprise IT
    ("INTU", "ADBE"),   # creative/financial SaaS
    ("CSCO", "JNPR"),   # networking equipment
    ("PANW", "FTNT"),   # cybersecurity
    ("ANET", "CSCO"),   # datacenter networking
    ("DELL", "HPE"),    # enterprise hardware
    ("CRWD", "FTNT"),   # cybersecurity / systems software [Top100]
    ("CRWD", "GEN"),    # systems software [Top100]
    ("FTNT", "GEN"),    # systems software [Top100]
    ("CSCO", "FFIV"),   # communications equipment [Top100]
    ("CSCO", "MSI"),    # communications equipment [Top100]
    ("FFIV", "MSI"),    # communications equipment [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # SEMICONDUCTORS & ELECTRONICS  (10 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("AMAT", "LRCX"),   # wafer fab equipment
    ("NVDA", "AMD"),    # GPU / AI chips
    ("AVGO", "QCOM"),   # mobile & broadband chips
    ("INTC", "TXN"),    # legacy chipmakers — analog/x86
    ("KLAC", "AMAT"),   # semi equipment cross
    ("MRVL", "AVGO"),   # datacenter semiconductors
    ("KLAC", "LRCX"),   # semi equipment [Top100]
    ("APH",  "COHR"),   # electronic components [Top100]
    ("COHR", "TEL"),    # electronic components [Top100]
    ("GLW",  "TEL"),    # electronic components [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # CONSUMER STAPLES  (10 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("KO",   "PEP"),    # beverages
    ("PG",   "CL"),     # household & personal care
    ("COST", "WMT"),    # warehouse vs mass retail
    ("MDLZ", "HSY"),    # snacks & confectionery
    ("MO",   "PM"),     # tobacco — domestic vs international
    ("KMB",  "CLX"),    # household products
    ("KDP",  "MNST"),   # soft drinks [Top100]
    ("KDP",  "PEP"),    # soft drinks [Top100]
    ("MNST", "PEP"),    # soft drinks [Top100]
    ("EL",   "KVUE"),   # personal care products [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # CONSUMER DISCRETIONARY  (17 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("HD",   "LOW"),    # home improvement
    ("MCD",  "QSR"),    # fast food
    ("TGT",  "WMT"),    # mass retail
    ("RCL",  "CCL"),    # cruise lines
    ("NKE",  "LULU"),   # athletic apparel
    ("SBUX", "MCD"),    # restaurants / QSR
    ("AZO",  "ORLY"),   # automotive retail [Top100]
    ("MGM",  "WYNN"),   # casinos & gaming [Top100]
    ("LVS",  "MGM"),    # casinos & gaming [Top100]
    ("LVS",  "WYNN"),   # casinos & gaming [Top100]
    ("NKE",  "RL"),     # apparel [Top100]
    ("LULU", "RL"),     # apparel [Top100]
    ("DHI",  "LEN"),    # homebuilding [Top100]
    ("DHI",  "PHM"),    # homebuilding [Top100]
    ("DHI",  "NVR"),    # homebuilding [Top100]
    ("COST", "DG"),     # consumer retail [Top100]
    ("COST", "DLTR"),   # consumer retail [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # HEALTHCARE  (14 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("UNH",  "CI"),     # health insurers — managed care
    ("MRK",  "PFE"),    # big pharma
    ("CVS",  "WBA"),    # pharmacy retail
    ("JNJ",  "ABT"),    # diversified healthcare / med devices
    ("LLY",  "NVO"),    # GLP-1 obesity/diabetes drugs
    ("AMGN", "GILD"),   # biotech — established
    ("SYK",  "MDT"),    # medical devices — ortho/surgical
    ("HCA",  "THC"),    # hospital operators
    ("CNC",  "HUM"),    # managed health care [Top100]
    ("CNC",  "UNH"),    # managed health care [Top100]
    ("ELV",  "HUM"),    # managed health care [Top100]
    ("ELV",  "UNH"),    # managed health care [Top100]
    ("COO",  "SOLV"),   # health care supplies [Top100]
    ("ALGN", "COO"),    # health care supplies [Top100]
    ("ALGN", "SOLV"),   # health care supplies [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # INDUSTRIALS  (18 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("DAL",  "UAL"),    # airlines
    ("UPS",  "FDX"),    # logistics
    ("LMT",  "RTX"),    # defense prime
    ("GD",   "NOC"),    # defense prime — C4ISR
    ("CSX",  "NSC"),    # eastern railroads
    ("CAT",  "DE"),     # heavy machinery
    ("BA",   "LMT"),    # aerospace & defense
    ("HON",  "MMM"),    # diversified industrials
    ("WM",   "RSG"),    # waste management
    ("GE",   "HON"),    # industrial conglomerates
    ("JBHT", "ODFL"),   # cargo ground transport [Top100]
    ("CSX",  "UNP"),    # rail transportation [Top100]
    ("NSC",  "UNP"),    # rail transportation [Top100]
    ("FAST", "URI"),    # trading companies [Top100]
    ("PAYX", "RHI"),    # HR & employment services [Top100]
    ("ADP",  "PAYX"),   # HR & employment services [Top100]
    ("ADP",  "RHI"),    # HR & employment services [Top100]
    ("DAL",  "LUV"),    # passenger airlines [Top100]
    ("DAL",  "AAL"),    # passenger airlines [Top100]
    ("LUV",  "UAL"),    # passenger airlines [Top100]
    ("LUV",  "AAL"),    # passenger airlines [Top100]
    ("UAL",  "AAL"),    # passenger airlines [Top100]
    ("RSG",  "ROL"),    # environmental services [Top100]
    ("ROL",  "WM"),     # environmental services [Top100]
    ("CTAS", "CPRT"),   # diversified support services [Top100]
    ("CTAS", "LDOS"),   # diversified support services [Top100]
    ("CPRT", "LDOS"),   # diversified support services [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # TELECOM & MEDIA  (4 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("T",    "VZ"),     # telecom
    ("DIS",  "CMCSA"),  # media + distribution duopoly
    ("NFLX", "DIS"),    # streaming
    ("CHTR", "CMCSA"),  # cable operators

    # ═══════════════════════════════════════════════════════════════════
    # AUTOS  (3 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("F",    "GM"),     # US automakers
    ("TSLA", "F"),      # EV vs legacy auto
    ("RIVN", "LCID"),   # pure-play EV startups

    # ═══════════════════════════════════════════════════════════════════
    # MATERIALS & MINING  (10 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("GOLD", "NEM"),    # gold miners — Barrick vs Newmont
    ("NEM",  "AEM"),    # gold miners — Newmont vs Agnico Eagle
    ("FCX",  "SCCO"),   # copper miners
    ("NUE",  "STLD"),   # steel producers
    ("APD",  "LIN"),    # industrial gases
    ("ADM",  "BG"),     # agricultural products [Top100]
    ("CTVA", "MOS"),    # fertilizers [Top100]
    ("CF",   "CTVA"),   # fertilizers [Top100]
    ("CF",   "MOS"),    # fertilizers [Top100]
    ("CRH",  "MLM"),    # construction materials [Top100]
    ("CRH",  "VMC"),    # construction materials [Top100]
    ("MLM",  "VMC"),    # construction materials [Top100]
    ("AMCR", "IP"),     # packaging [Top100]
    ("AMCR", "PKG"),    # packaging [Top100]
    ("AVY",  "IP"),     # packaging [Top100]

    # ═══════════════════════════════════════════════════════════════════
    # REITs  (7 pairs)
    # ═══════════════════════════════════════════════════════════════════
    ("PLD",  "SPG"),    # logistics vs retail REITs
    ("AMT",  "CCI"),    # cell tower REITs
    ("O",    "NNN"),    # net lease REITs
    ("PSA",  "EXR"),    # self-storage REITs
    ("DLR",  "EQIX"),   # data center REITs [Top100]
    ("ARE",  "BXP"),    # office REITs [Top100]
    ("DOC",  "WELL"),   # health care REITs [Top100]
]

TICKERS = list(dict.fromkeys(t for pair in PAIRS for t in pair))

START_DATE   = "2023-01-01"   # Polygon intraday start (Starter plan ~2 years)
END_DATE     = "2026-04-28"
DAILY_START  = "2015-01-01"   # Yahoo daily start — 10 years for pair selection

REQUEST_SLEEP = 25

RTH_START = "09:30"
RTH_END = "16:00"
SIGNAL_START = "10:00"

BAR_MINUTES  = 15                             # 15-min bars from Polygon
BARS_PER_DAY = int(6.5 * 60 / BAR_MINUTES)   # 26 for 15-min

CLOSES_FILE  = f"closes_{BAR_MINUTES}min.csv"
VOLUMES_FILE = f"volumes_{BAR_MINUTES}min.csv"
VWAPS_FILE   = f"vwaps_{BAR_MINUTES}min.csv"

RECENT_BARS  = BARS_PER_DAY * 92             # ~4.6 months regardless of bar size
TRAIN_RATIO  = 0.55   # first 55% → find pairs; last 45% → out-of-sample test

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

KALMAN_DELTA = 3e-5  # process noise — controls how fast beta adapts (1/delta ≈ adaptation window in bars)
# 3e-5 → beta adapts over ~33,000 bars (~425 days at 15-min, ~142 days at 5-min)
# 1e-4 → adapts over ~10,000 bars (too fast — kills mean-reversion signal)

HURST_MAX = 0.50    # max Hurst exponent for pair spread (< 0.5 = mean reverting)
CORR_MIN  = 0.50    # min log-return correlation over training window
RECENT_CORR_DAYS = 120   # rolling window for recent correlation check (calendar days)
RECENT_CORR_MIN  = 0.50  # pair disabled if recent 120-day correlation drops below this

# ── Phase 1: rolling window cointegration ────────────────────────────────────
COINT_WINDOW_DAYS  = 90    # rolling window for EG cointegration test (trading days)
COINT_BREAK_P      = 0.15  # if rolling coint p > this during backtest → suspend pair
COINT_RECHECK_DAYS = 5     # recheck coint every N trading days during backtest

# ── Phase 4: K-Means macro regime ────────────────────────────────────────────
KMEANS_N_CLUSTERS = 3      # 0=Trend, 1=Sideways, 2=Panic
KMEANS_VOL_WINDOW = 20     # rolling window for macro features (trading days)

PAIR_MAX_LOSS = -30.0      # disable pair if cumulative net P&L drops below this

IV_LOOKBACK   = 60         # days for IV percentile calculation
IV_THRESHOLD  = 75         # percentile above which → reduce position size
IV_SIZE_HIGH  = 0.5        # position size when IV is elevated
IV_SIZE_NORM  = 1.0        # position size when IV is normal

# Dynamic position sizing (step9)
REGIME_MULT_NORMAL   = 1.0   # full size in normal regime
REGIME_MULT_VOLATILE = 0.3   # reduced size in volatile regime
IV_MULT_MAX          = 1.0   # full size when IV is at its lowest
IV_MULT_MIN          = 0.5   # half size when IV is at its highest
MIN_POSITION_SIZE    = 0.15  # skip trade entirely if combined size below this

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
