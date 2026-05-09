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
    # ("EL", "KVUE"),   # KVUE IPO 2023 — insufficient history

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
    # ("COO",  "SOLV"),  # SOLV IPO 2024 — insufficient history
    ("ALGN", "COO"),    # health care supplies [Top100]
    # ("ALGN","SOLV"),  # SOLV IPO 2024 — insufficient history

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
    # ("RIVN","LCID"),  # RIVN/LCID IPO 2021 — insufficient history

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

    # ═══════════════════════════════════════════════════════════════════
    # UNIVERSE DISCOVERY — pairs_universe.py (Johansen, joh_margin > 2.5)
    # ═══════════════════════════════════════════════════════════════════

    # Already in intraday data:
    ("AVGO", "NVDA"),  # Semiconductors            joh_margin=9.90  H=0.052
    ("CRWD", "PANW"),  # Systems Software           joh_margin=3.11  H=0.160
    ("GD",   "RTX"),   # Aerospace & Defense        joh_margin=3.06  H=0.178
    ("LVS",  "WYNN"),  # Casinos & Gaming           joh_margin=2.96  H=0.152
    ("COP",  "EOG"),   # Oil & Gas E&P              joh_margin=2.55  H=0.099

    # Needs intraday download:
    ("ACGL", "HIG"),   # Property & Casualty Ins.   joh_margin=14.45 H=0.160
    ("ADSK", "INTU"),  # Application Software       joh_margin=7.95  H=0.203
    ("MDT",  "RMD"),   # Health Care Equipment      joh_margin=6.58  H=0.189
    # ("CTSH","EPAM"),  # EPAM — Ukraine war disruption 2022, unstable behavior
    ("GD",   "LHX"),   # Aerospace & Defense        joh_margin=4.04  H=0.112
    ("IFF",  "PPG"),   # Specialty Chemicals        joh_margin=3.84  H=0.358
    ("FIS",  "JKHY"),  # Payment Processing         joh_margin=3.55  H=0.371
    ("BMY",  "MRK"),   # Pharmaceuticals            joh_margin=3.27  H=0.198
    ("FDS",  "SPGI"),  # Financial Exchanges & Data joh_margin=2.71  H=0.251
    ("CAH",  "MCK"),   # Health Care Distributors   joh_margin=2.25  H=0.297

    # --- NEW MINED PAIRS (Alpha Vantage) ---
    ("PPL", "WEC"),
    ("D", "XLU"),
    ("DTE", "DUK"),
    ("JPM", "XLC"),
    ("IR", "PPG"),
    ("EQR", "ESS"),
    ("ADSK", "PANW"),
    ("ESS", "UDR"),
    ("ADSK", "MCO"),
    ("MTB", "SNA"),
    ("CPT", "ESS"),
    ("IQV", "TMO"),
    ("AMP", "CPAY"),
    ("FITB", "ITW"),
    ("AVB", "ESS"),
    ("ESS", "MAA"),
    ("MCO", "MET"),
    ("ALL", "TRV"),
    ("IR", "PRU"),
    ("AMT", "CCI"),
    ("AEE", "PPL"),
    ("GOOG", "GOOGL"),
    ("DTE", "SO"),
    ("MTD", "TMO"),
    ("MGM", "TROW"),
    ("AFL", "ALL"),
    ("FITB", "USB"),
    ("ESS", "EXR"),
    ("CPB", "GIS"),
    ("MA", "MCO"),
    ("BX", "KKR"),
    ("ARES", "MCO"),
    ("SPG", "TFC"),
    ("CMS", "DUK"),
    ("AME", "MAR"),
    ("ALL", "CINF"),
    ("AMP", "APO"),
    ("CAT", "KLAC"),
    ("AEE", "NI"),
    ("AVY", "IR"),
    ("EQR", "EXR"),
    ("WSM", "XLY"),
    ("AMP", "MCO"),
    ("FITB", "SPG"),
    ("AMT", "AWK"),
    ("IEX", "ODFL"),
    ("MCHP", "NXPI"),
    ("ESS", "PSA"),
    ("BRO", "MRSH"),
    ("DOV", "JBHT"),
    ("ICE", "MCO"),
    ("ATO", "EVRG"),
    ("CLX", "PG"),
    ("DUK", "SO"),
    ("FITB", "PNC"),
    ("HON", "ITW"),
    ("CTSH", "JKHY"),
    ("MCO", "PNR"),
    ("PNC", "SPG"),
    ("SPG", "USB"),
    ("ITW", "LOW"),
    ("NDAQ", "XLF"),
    ("FITB", "KEY"),
    ("BX", "MCO"),
    ("AXP", "CRH"),
    ("ADSK", "TRMB"),
    ("EQR", "UDR"),
    ("HD", "LII"),
    ("LOW", "PKG"),
    ("KEY", "USB"),
    ("HD", "SHW"),
    ("ATO", "DUK"),
    ("AFL", "AIZ"),
    ("ROK", "XLK"),
    ("IFF", "PPG"),
    ("LII", "SHW"),
    ("CPT", "UDR"),
    ("ES", "FE"),
    ("BBY", "HPQ"),
    ("HUBB", "ITW"),
    ("MSCI", "XLF"),
    ("SYY", "XLP"),
    ("CMS", "SO"),
    ("JBHT", "UNP"),
    ("CMS", "PPL"),
    ("PPL", "XEL"),
    ("ECL", "ITW"),
    ("BAC", "CRH"),
    ("ATO", "XEL"),
    ("BAC", "XLC"),
    ("DOV", "UNP"),
    ("ALL", "HIG"),
    ("ITW", "PKG"),
    ("CPT", "EQR"),
    ("DTE", "EXC"),
    ("DUK", "PPL"),
    ("DOC", "EQR"),
    ("DTE", "PPL"),
    ("LNT", "NI"),
    ("MTD", "SWK"),
    ("PNC", "SNA"),
    ("BAC", "EMR"),
    ("TFC", "USB"),
    ("COF", "WFC"),
    ("MCO", "V"),
    ("A", "GEHC"),
    ("PPG", "UPS"),
    ("DHR", "RVTY"),
    ("BALL", "PPG"),
    ("AXP", "MCO"),
    ("CPT", "EXR"),
]

TICKERS = list(dict.fromkeys(t for pair in PAIRS for t in pair))

START_DATE   = "2006-01-01"   # Alpha Vantage intraday start (20 years history)
END_DATE     = "2026-04-28"
DAILY_START  = "2006-01-01"   # Yahoo daily start — full 20 years for pair selection

REQUEST_SLEEP = 25

RTH_START = "09:30"
RTH_END = "16:00"
SIGNAL_START = "09:30"  # Opened up to catch Price Discovery (Morning Gaps)

BAR_MINUTES  = 15                             # 15-min bars from Alpha Vantage
BARS_PER_DAY = int(6.5 * 60 / BAR_MINUTES)   # 26 for 15-min

CLOSES_FILE  = f"closes_{BAR_MINUTES}min.csv"
VOLUMES_FILE = f"volumes_{BAR_MINUTES}min.csv"
VWAPS_FILE   = f"vwaps_{BAR_MINUTES}min.csv"

RECENT_BARS  = BARS_PER_DAY * 92             # ~4.6 months regardless of bar size
TRAIN_RATIO  = 0.70   # first 70% → find pairs; last 30% → out-of-sample test

CORR_THRESHOLD = 0.5
CORR_TOP_N = 10
COINT_TOP_N = 3

ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 3.5
ENTRY_Z_VOLATILE = 2.8   # stricter threshold when HMM detects volatile regime

# ── Regime-Conditioned Profiling (RCDP) ─────────────────────────────────────
# Wider grids for per-regime grid search (regime_profiler.py)
RCDP_ENTRY_GRID  = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
RCDP_EXIT_GRID   = [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
RCDP_STOP_GRID   = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0]
RCDP_MIN_TRADES_VOLATILE = 5   # volatile slice has fewer bars — lower threshold

# Transaction costs (per side, per leg, as fraction of price)
COST_COMMISSION = 0.0003   # broker commission (e.g. IBKR tiered)
COST_SPREAD     = 0.0003   # half of bid-ask spread (liquid large-caps)
COST_SLIPPAGE   = 0.0001   # slippage: signal on close, fill near open

# Passive Aggressor: Limit orders inside spread for entries and TP (earn spread/rebate)
COST_MAKER      = COST_COMMISSION                           # 0.03%
# Taker: Market orders for stop-loss and macro panics
COST_TAKER      = COST_COMMISSION + COST_SPREAD + COST_SLIPPAGE  # 0.07%

# ── Idiosyncratic Circuit Breaker ────────────────────────────────────────────
CIRCUIT_BREAKER_Z = 4.5  # Max Z-score before immediate hard-stop and pair block

BORROW_RATE_ANNUAL = 0.015  # 1.5% annual short borrow (liquid stocks)

KALMAN_DELTA = 3e-6  # process noise — controls how fast beta adapts (1/delta ≈ adaptation window in bars)
# 3e-6 → beta adapts over ~333,000 bars (~12,800 trading days) — near-fixed beta
# 3e-5 → adapts over ~33,000 bars (too fast — innovations become white noise)
# 1e-4 → adapts over ~10,000 bars (way too fast — kills mean-reversion signal)

HALF_LIFE_MAX_BARS = 500  # filter out pairs slower than ~20 trading days (spread too sluggish)

HURST_MAX = 0.50    # max Hurst exponent for pair spread (< 0.5 = mean reverting)

# ── Hurst Entry Gate (dynamic, per-trade) ────────────────────────────────────
# Computed lazily on the DAILY spread when |z| >= entry threshold.
# Blocks entries when the spread is trending (structural drift), regardless of macro regime.
HURST_ENTRY_WINDOW = 60     # daily bars of spread history for rolling Hurst at entry time
HURST_ENTRY_MAX    = 0.55   # block entry if H > this (spread is trending, not stretching)
CORR_MIN  = 0.50    # min log-return correlation over training window
RECENT_CORR_DAYS = 120   # rolling window for recent correlation check (calendar days)
RECENT_CORR_MIN  = 0.50  # pair disabled if recent 120-day correlation drops below this

# ── Gatev et al. SSD pre-filter (pairs_universe.py Layer 2b) ───────────────────
# Sum of Squared Deviations of normalised cumulative return indices.
# Keeps only the X% of pairs with smallest SSD — fast pre-filter before Johansen.
SSD_PERCENTILE = 80   # keep bottom 80% by SSD (discard top 20% most divergent)

# ── Phase 1: rolling window cointegration ────────────────────────────────────
COINT_WINDOW_DAYS  = 90    # rolling window for EG cointegration test (trading days)
COINT_BREAK_P      = 0.15  # if rolling coint p > this during backtest → suspend pair
COINT_RECHECK_DAYS = 5     # recheck coint every N trading days during backtest

# ── Phase 4: K-Means macro regime ────────────────────────────────────────────
KMEANS_N_CLUSTERS = 3      # 0=Trend, 1=Sideways, 2=Panic
KMEANS_VOL_WINDOW = 20     # rolling window for macro features (trading days)

# ── Walk-Forward Optimization (WFO) ───────────────────────────────────────
# Gatev et al. (2006) baseline: 12-month train, 6-month OOS, 6-month step
WFO_TRAIN_MONTHS = 12   # formation window (months)
WFO_TEST_MONTHS  = 6    # trading window / OOS test (months)
WFO_STEP_MONTHS  = 6    # how far each window slides forward
WFO_MIN_TRADES   = 5    # minimum trades per OOS window to count as valid
WFO_EXPANDING    = True  # True = expanding window (train_start anchored to data origin)
                         # False = rolling window (classic Gatev fixed-width train)

PAIR_MAX_LOSS = -30.0      # disable pair if cumulative net P&L drops below this

IV_LOOKBACK   = 60         # days for IV percentile calculation
IV_THRESHOLD  = 75         # percentile above which → reduce position size
IV_SIZE_HIGH  = 0.5        # position size when IV is elevated
IV_SIZE_NORM  = 1.0        # position size when IV is normal

# Dynamic position sizing (step9)
REGIME_MULT_NORMAL   = 1.0   # full size in normal regime
REGIME_MULT_VOLATILE = 0.3   # reduced size in volatile regime (per-pair HMM)
HMM_PANIC_MULT       = 0.333 # global macro HMM panic: cut ALL sizes by 3
IV_MULT_MAX          = 1.0   # full size when IV is at its lowest
IV_MULT_MIN          = 0.5   # half size when IV is at its highest
MIN_POSITION_SIZE    = 0.15  # skip trade entirely if combined size below this

INITIAL_CAPITAL = 10_000        # starting portfolio balance in USD
ALLOCATION_METHOD = "markowitz"    # "equal" | "sharpe" | "markowitz" (MVO)
MAX_PAIR_WEIGHT   = 0.15        # cap: no single pair gets more than 15% of capital

TARGET_RISK_USD = 150.0         # volatility scaling: target dollar risk per trade (1σ of spread)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# ── The Black Swan Hedge (Tail Risk Convexity) ───────────────────────────────
TAIL_HEDGE_DRAG_ANNUAL = 0.015  # 1.5% annual drag on portfolio (buying far OTM Puts)
TAIL_HEDGE_PAYOUT_MULT = 10.0   # Convexity multiplier when HMM detects Panic
