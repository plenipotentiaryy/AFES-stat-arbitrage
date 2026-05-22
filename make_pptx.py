"""Build AFES_Presentation.pptx matching the AFES_Club_Introduction style."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette (extracted from sample) ─────────────────────────────────────────
NAVY    = RGBColor(0x0D, 0x1B, 0x30)
LBGD    = RGBColor(0xF2, 0xF5, 0xFC)
GOLD    = RGBColor(0xE8, 0xB8, 0x4B)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1B, 0x2A, 0x4A)
LBLUE   = RGBColor(0xCA, 0xDC, 0xFC)
MUTED   = RGBColor(0x7A, 0x8F, 0xA6)
MID     = RGBColor(0x2E, 0x5B, 0x8A)
MID2    = RGBColor(0x3A, 0x68, 0x98)

I = Inches

# ── Low-level helpers ────────────────────────────────────────────────────────

def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color

def rect(slide, l, t, w, h, color, line=False):
    s = slide.shapes.add_shape(1, I(l), I(t), I(w), I(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if not line:
        s.line.fill.background()
    return s

def txb(slide, l, t, w, h):
    box = slide.shapes.add_textbox(I(l), I(t), I(w), I(h))
    box.text_frame.word_wrap = True
    return box.text_frame

def para(tf, text, font='Calibri', size=12, bold=False, color=WHITE,
         align=PP_ALIGN.LEFT, first=False, space_before=0, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = Pt(space_before)
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return p

# ── Composite helpers ────────────────────────────────────────────────────────

def tag(slide, text, dark_slide=False):
    """Small coloured label at top-left."""
    fill = MID if not dark_slide else DARK
    rect(slide, 0.5, 0.12, 2.6, 0.27, fill)
    tf = txb(slide, 0.5, 0.12, 2.6, 0.27)
    para(tf, text, size=9.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, first=True)

def slide_title(slide, text, dark=False):
    """Gold-stripe + Georgia title."""
    rect(slide, 0.5, 0.52, 0.07, 0.72, GOLD)
    c = WHITE if dark else DARK
    tf = txb(slide, 0.68, 0.52, 9.0, 0.72)
    para(tf, text, font='Georgia', size=28, bold=True, color=c, first=True)

def card(slide, l, t, w, h, title, body,
         bg_color=DARK, title_color=GOLD, body_color=LBLUE,
         title_size=13, body_size=10.5, accent=True):
    """Dark card with gold-accent stripe, title, body."""
    rect(slide, l, t, w, h, bg_color)
    if accent:
        rect(slide, l, t, 0.07, h, GOLD)
    # title
    tf = txb(slide, l + 0.2, t + 0.14, w - 0.28, 0.38)
    para(tf, title, font='Georgia', size=title_size, bold=True, color=title_color, first=True)
    # body
    tf2 = txb(slide, l + 0.2, t + 0.56, w - 0.28, h - 0.66)
    para(tf2, body, size=body_size, color=body_color, first=True)

def stat_block(slide, l, t, value, label):
    """Large gold stat + small label (dark-slide style)."""
    tf = txb(slide, l, t, 2.2, 0.72)
    para(tf, value, font='Georgia', size=36, bold=True, color=GOLD, first=True)
    tf2 = txb(slide, l, t + 0.76, 2.2, 0.42)
    para(tf2, label, size=10, color=LBLUE, first=True)

# ═══════════════════════════════════════════════════════════════════════════
# BUILD PRESENTATION
# ═══════════════════════════════════════════════════════════════════════════

prs = Presentation()
prs.slide_width  = I(10)
prs.slide_height = I(5.625)
BLK = prs.slide_layouts[6]   # blank

# ── SLIDE 1 — Title (dark, matches sample slide 1) ──────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
rect(s, 0.0, 0.0, 0.22, 5.625, GOLD)          # left gold stripe
rect(s, 0.0, 4.85, 10.0, 0.775, DARK)          # bottom bar

tf = txb(s, 0.42, 0.75, 9.2, 0.4)
para(tf, 'QUANTITATIVE FINANCE CLUB', size=11, bold=True, color=GOLD, first=True)

tf = txb(s, 0.42, 1.25, 9.0, 2.0)
para(tf, 'AFES — Statistical Arbitrage', font='Georgia', size=40, bold=True, color=WHITE, first=True)
para(tf, 'Engine', font='Georgia', size=40, bold=True, color=WHITE)

tf = txb(s, 0.42, 3.35, 9.0, 0.55)
para(tf, 'Step-by-Step: From Pair Discovery to Live Trading Signal', size=19, color=LBLUE, first=True)

tf = txb(s, 0.42, 4.92, 9.2, 0.5)
para(tf, 'Quantitative Finance Club  ·  Kraków  ·  2026', size=11, color=MUTED, first=True)

# ── SLIDE 2 — Agenda (light, 2×2 grid) ──────────────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, "TODAY'S AGENDA")
slide_title(s, 'What We Cover Today')

# 2×2 agenda cards  (wider than sample's 6-item because we have 4)
agenda = [
    ('01', 'Step 2 — Finding Pairs',      'Cointegration testing on 324 candidate pairs to find the ones with a genuine statistical relationship.'),
    ('02', 'Step 3 — 10 Safety Filters',  'Ten independent conditions that must all pass before any trade is placed.'),
    ('03', 'Step 4 — The Backtest',        'Replaying 5 years of prices, every 5 minutes, with real costs and all filters active.'),
    ('04', 'Step 5 — Proving It\'s Real', 'Six independent verification tests to confirm the results aren\'t luck.'),
]
positions = [(0.5, 1.38), (5.1, 1.38), (0.5, 3.18), (5.1, 3.18)]
for (num, title, desc), (lx, ty) in zip(agenda, positions):
    rect(s, lx, ty, 4.4, 1.62, DARK)
    tf = txb(s, lx + 0.14, ty + 0.12, 0.55, 1.2)
    para(tf, num, font='Georgia', size=26, bold=True, color=GOLD, first=True)
    tf2 = txb(s, lx + 0.78, ty + 0.14, 3.46, 0.42)
    para(tf2, title, font='Georgia', size=13, bold=True, color=WHITE, first=True)
    tf3 = txb(s, lx + 0.78, ty + 0.60, 3.46, 0.9)
    para(tf3, desc, size=9.5, color=LBLUE, first=True)

# ── SLIDE 3 — Core Concept (dark hero) ──────────────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
rect(s, 0.0, 0.0, 0.22, 5.625, GOLD)

tf = txb(s, 0.5, 1.1, 9.0, 0.42)
para(tf, 'THE CORE IDEA', size=11, bold=True, color=GOLD, first=True)

tf = txb(s, 0.5, 1.6, 9.0, 1.3)
para(tf, 'The Dog & The Leash', font='Georgia', size=54, bold=True, color=WHITE, first=True)

tf = txb(s, 0.5, 3.05, 9.0, 0.52)
para(tf, 'Statistical arbitrage in one image — and why it works', size=18, color=LBLUE, first=True)

tf = txb(s, 0.5, 3.7, 9.0, 0.7)
para(tf, 'Everything in this presentation flows from this single analogy.', size=13, italic=True, color=MUTED, first=True)

# ── SLIDE 4 — The Analogy (light, 3 cards) ──────────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'THE CORE ANALOGY')
slide_title(s, 'A Person, A Dog, A Leash')

card(s, 0.5,  1.55, 2.85, 3.55,
     'The Person',
     'JPMorgan. Walks at a steady pace. Sets the rhythm. Anchors the relationship.',
     bg_color=DARK, body_color=LBLUE)

card(s, 3.58, 1.55, 2.85, 3.55,
     'The Dog',
     'Bank of America. Runs ahead. Falls behind. Sniffs around. But always comes back.',
     bg_color=MID, body_color=LBLUE)

card(s, 6.66, 1.55, 2.85, 3.55,
     'The Leash',
     'The same industry. Same customers. Same news. Same regulators. The economics that always pull them back together.',
     bg_color=MID2, body_color=LBLUE)

tf = txb(s, 0.5, 5.22, 9.0, 0.28)
para(tf, 'When the gap gets too wide → we bet on the snap-back.  We profit from the relationship, not the market direction.',
     size=10, italic=True, color=MUTED, first=True)

# ── SLIDE 5 — Step 2 Section (dark, content style) ──────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
tag(s, 'STEP 02', dark_slide=True)

rect(s, 0.5, 0.52, 0.07, 0.72, GOLD)
tf = txb(s, 0.68, 0.52, 9.0, 0.72)
para(tf, 'Finding the Right Pairs', font='Georgia', size=28, bold=True, color=WHITE, first=True)

tf = txb(s, 0.68, 1.38, 8.8, 0.56)
para(tf, '324 candidate pairs. Statistical tests. Only 20–40 survive with a proven leash.', size=13, color=LBLUE, first=True)

# Three fact rows
facts = [
    ('Engle-Granger',  'p < 0.15 required — the gap reliably closes. Pull the rope; if the dog snaps back, the leash is real.'),
    ('Johansen Test',  'Second, stricter confirmation. A pair passes if either test succeeds. Two independent witnesses.'),
    ('The Survivors',  'Each pair also provides beta (the hedge ratio) — how many shares of Stock B to short per share of Stock A.'),
]
for i, (title, body) in enumerate(facts):
    ty = 2.08 + i * 0.98
    tf = txb(s, 0.7, ty, 4.0, 0.32)
    para(tf, title, font='Georgia', size=12, bold=True, color=GOLD, first=True)
    tf2 = txb(s, 0.7, ty + 0.34, 8.8, 0.52)
    para(tf2, body, size=10.5, color=LBLUE, first=True)

# ── SLIDE 6 — Cointegration detail (light, 3 cards) ─────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'STEP 02 — PAIR SELECTION')
slide_title(s, 'The Cointegration Test')

card(s, 0.5, 1.55, 2.85, 3.55,
     'What We Measure',
     'Subtract one stock from the other (scaled by beta). Is the result mean-reverting? Does the gap always come back to zero?',
     bg_color=DARK)

card(s, 3.58, 1.55, 2.85, 3.55,
     'Two Independent Tests',
     'Engle-Granger: simple OLS-based test.\nJohansen: multivariate, more robust.\nA pair passes if EITHER confirms the relationship.',
     bg_color=MID)

card(s, 6.66, 1.55, 2.85, 3.55,
     'Three Extra Filters',
     'Hurst < 0.50: gap shrinks, not drifts.\nCorrelation > 0.50: move together.\nHalf-Life: snaps back in days or weeks, not years.',
     bg_color=MID2)

tf = txb(s, 0.5, 5.22, 9.0, 0.28)
para(tf, 'Result: pairs_selected.csv — the confirmed, mathematically-proven leash pairs that drive all downstream steps.',
     size=10, italic=True, color=MUTED, first=True)

# ── SLIDE 7 — Extra Filters Stats (dark, slide-12 style) ────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
tag(s, 'STEP 02 — QUALITY GATES', dark_slide=True)
rect(s, 0.5, 0.52, 0.07, 0.72, GOLD)
tf = txb(s, 0.68, 0.52, 9.0, 0.72)
para(tf, 'Three Quality Gates Every Pair Must Pass', font='Georgia', size=28, bold=True, color=WHITE, first=True)

stat_block(s, 0.5,  1.56, '< 0.50',   'Hurst exponent — gap must shrink, not drift')
stat_block(s, 3.0,  1.56, '> 0.50',   'Minimum correlation — they must move together')
stat_block(s, 5.5,  1.56, '< 40d',    'Half-life — must snap back in days, not months')
stat_block(s, 8.0,  1.56, '20–40',    'Pairs surviving out of 324 tested')

tf = txb(s, 0.5, 2.9, 9.0, 1.8)
para(tf, 'The Hurst Exponent', font='Georgia', size=12, bold=True, color=GOLD, first=True)
para(tf, 'A number between 0 and 1. Below 0.5 means the gap behaves like a rubber band — stretched, it snaps back. Above 0.5 means it drifts like a spring — once it unwinds, it keeps going. We require < 0.50.')
para(tf, 'Half-Life', font='Georgia', size=12, bold=True, color=GOLD, space_before=6)
para(tf, 'How long until the gap closes by half? Even with a real leash, if it takes 3 years to close, we cannot hold the trade — transaction costs and borrow fees will consume the profit long before resolution.')

# ── SLIDE 8 — Step 3 Section (dark hero) ────────────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
rect(s, 0.0, 0.0, 0.22, 5.625, GOLD)

tf = txb(s, 0.5, 1.1, 9.0, 0.42)
para(tf, 'THE SAFETY SYSTEM', size=11, bold=True, color=GOLD, first=True)

tf = txb(s, 0.5, 1.55, 2.0, 1.5)
para(tf, '10', font='Georgia', size=96, bold=True, color=WHITE, first=True)

tf = txb(s, 2.6, 1.7, 7.0, 1.2)
para(tf, 'Safety Filters', font='Georgia', size=48, bold=True, color=WHITE, first=True)

tf = txb(s, 0.5, 3.3, 9.0, 0.52)
para(tf, 'All 10 must pass simultaneously before a single trade is placed.', size=18, color=LBLUE, first=True)

tf = txb(s, 0.5, 3.95, 9.0, 0.38)
para(tf, 'Think of them as 10 locked doors — the trade unlocks each one or it never gets through.', size=13, color=MUTED, first=True)

# ── SLIDE 9 — Layers 1-2: Regime (light, 2 big cards) ───────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'STEP 03 — LAYERS 1 & 2')
slide_title(s, 'Regime Detection: Per-Pair & Market-Wide')

card(s, 0.5, 1.55, 4.4, 3.55,
     'Layer 1 — HMM Regime (per pair)',
     'Hidden Markov Model watches each pair individually.\n\nAsks: is this specific dog calm or erratic today?\n\n▸ State 0 (Normal): spread moves predictably → trade at standard thresholds\n▸ State 1 (Volatile): spread is jumpy → require bigger gap (Z = 2.8 vs 2.0) before entering\n\nAnalogy: Check your dog\'s mood before each walk.',
     bg_color=DARK, title_size=13, body_size=10)

card(s, 5.1, 1.55, 4.4, 3.55,
     'Layer 2 — K-Means Macro (market-wide)',
     'Classifies the whole market into 3 states using SPY + VIX data.\n\n▸ Sideways: calm, range-bound → best environment, full trading\n▸ Trend: directional move → trade with caution\n▸ Panic: crash or spike → NO new entries; force-close existing trades\n\nAnalogy: Check the citywide weather before walking any dogs.',
     bg_color=MID, title_size=13, body_size=10)

# ── SLIDE 10 — Layers 3-4: Risk (light, 2 big cards) ────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'STEP 03 — LAYERS 3 & 4')
slide_title(s, 'Risk Assessment: Volatility & Simulation')

card(s, 0.5, 1.55, 4.4, 3.55,
     'Layer 3 — Implied Volatility (VIX)',
     'How nervous are other investors right now?\n\n▸ VIX < 15: calm → full position (1.0×)\n▸ VIX 15–25: normal → standard trading\n▸ VIX > 25th pct: elevated → half position (0.5×)\n▸ VIX > 90th pct: extreme → skip entirely\n\nAlso checks VIX9D/VIX ratio: when near-term fear > long-term fear, something specific is coming.\n\nAnalogy: How nervous are the other dog owners in the park?',
     bg_color=DARK, title_size=13, body_size=9.5)

card(s, 5.1, 1.55, 4.4, 3.55,
     'Layer 4 — OU Monte Carlo (10,000 paths)',
     'Simulate the spread\'s future 10,000 times using the Ornstein-Uhlenbeck model.\n\n▸ Estimate: θ (reversion speed), μ (equilibrium), σ (noise)\n▸ Run 10,000 simulated trades — enter, exit, stop, pay costs\n▸ Measure: win rate, VaR, CVaR, expected P&L\n\nIf expected P&L is negative → don\'t trade this pair regardless.\n\nAnalogy: Simulate 10,000 possible walks in your head before deciding to go.',
     bg_color=MID, title_size=13, body_size=9.5)

# ── SLIDE 11 — Layers 5-10: Sizing + Optimisation (light, 3 cards) ──────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'STEP 03 — LAYERS 5–10')
slide_title(s, 'Position Sizing & Parameter Optimisation')

card(s, 0.5, 1.55, 2.85, 3.55,
     'Layer 5 — Dynamic Sizing',
     'Combines all previous layers:\n\nsize = HMM × IV × MC × Macro\n\nAll multiply together. If any is near zero, the position collapses. If Macro = 0 (Panic), nothing trades — regardless of everything else.\n\nTarget: constant dollar risk per trade across all pairs.',
     bg_color=DARK, body_size=10)

card(s, 3.58, 1.55, 2.85, 3.55,
     'Layers 6–7: Grid & RCDP',
     'Grid (3f): Test 18 combos of (entry Z, exit Z, stop Z) on training data. Pick best per pair.\n\nRCDP (3g): Run grid separately for calm and volatile regimes. Each regime gets its own optimal thresholds.\n\nResult: entry at Z=1.8 for JPM-BAC, Z=2.2 for NVDA-AMD.',
     bg_color=MID, body_size=10)

card(s, 6.66, 1.55, 2.85, 3.55,
     'Layers 8–10: Profilers + WFO',
     'Profilers (3h/3i): Find entry Z where historical reward/risk > 1.3×.\n\nWalk-Forward (3j): Train 12m → test 6m → roll forward. Repeat across full history.\n\nThis is the hardest test: parameters found in one period must work in the next period never seen before.',
     bg_color=MID2, body_size=10)

# ── SLIDE 12 — Step 4 Section (dark, content) ───────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
tag(s, 'STEP 04', dark_slide=True)
rect(s, 0.5, 0.52, 0.07, 0.72, GOLD)
tf = txb(s, 0.68, 0.52, 9.0, 0.72)
para(tf, 'The Backtest', font='Georgia', size=28, bold=True, color=WHITE, first=True)

tf = txb(s, 0.68, 1.38, 8.8, 0.56)
para(tf, 'Replay 5 years of prices, every 5 minutes, with all filters active — and count every dollar.', size=13, color=LBLUE, first=True)

rows = [
    ('Time Machine',    'We pretend it is 2020 and execute the strategy forward in time. Every 5-minute bar triggers the full decision loop.'),
    ('Zero Look-Ahead', 'Pair selection used old data. The backtest only trades on data the model has never seen — the last 30% of the timeline.'),
    ('Real Costs',      'Commission + bid-ask spread + slippage on entry/exit. Daily borrow fee on every short position. No free lunch.'),
    ('The Output',      'Total return, Sharpe ratio, win rate, max drawdown, per-pair breakdown — and a comparison against SPY buy-and-hold.'),
]
for i, (title, body) in enumerate(rows):
    ty = 2.08 + i * 0.82
    tf = txb(s, 0.7, ty, 2.4, 0.32)
    para(tf, title, font='Georgia', size=11, bold=True, color=GOLD, first=True)
    tf2 = txb(s, 3.2, ty, 6.3, 0.65)
    para(tf2, body, size=10, color=LBLUE, first=True)

# ── SLIDE 13 — Backtest Mechanics (light, 3 cards) ──────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'STEP 04 — THE ENGINE')
slide_title(s, 'How the Backtest Works Bar by Bar')

card(s, 0.5, 1.55, 2.85, 3.55,
     'The Decision Loop',
     'Every 5-minute bar:\n1. In a trade? Check exit conditions.\n2. Not in a trade? Check 7 entry gates.\n3. All gates pass? Enter on the NEXT bar (no look-ahead).\n\nGates: Z-score, coint validity, no panic, Hurst, KDE density, macro filter, min position size.',
     bg_color=DARK, body_size=9.5)

card(s, 3.58, 1.55, 2.85, 3.55,
     'The Kalman Filter',
     'The hedge ratio (beta) is not fixed — it updates every 5 minutes.\n\nLike a GPS tracker that re-estimates the true leash length at every step.\n\nThis prevents stale ratios from creating phantom gaps after an earnings shock or macro shift.',
     bg_color=MID, body_size=9.5)

card(s, 6.66, 1.55, 2.85, 3.55,
     'Three Exit Types',
     '▸ SIGNAL: Z-score reverts to EXIT_Z → take the profit\n▸ STOP: Z-score hits STOP_Z going wrong way → cut the loss\n▸ TIME_STOP: held too long without resolution → exit\n\nPlus two forced exits:\n▸ Circuit breaker at |Z| > 4.5\n▸ Macro Panic → force-close all pairs',
     bg_color=MID2, body_size=9.5)

# ── SLIDE 14 — Safeguards (light, 4 stat blocks) ────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'STEP 04 — SAFEGUARDS')
slide_title(s, 'Key Protections Built Into the Backtest')

card(s, 0.5,  1.55, 4.4, 1.62,
     'Static Z-Score Lock',
     'Once in a trade, the Kalman filter is frozen. Exit/stop decisions use the Z-score from the moment of entry — the filter cannot adapt away a real loss.',
     bg_color=DARK, body_size=10)

card(s, 5.1,  1.55, 4.4, 1.62,
     'Circuit Breaker — |Z| > 4.5',
     'If the gap blows past 4.5 standard deviations mid-trade, something fundamental changed. Emergency close. Pair suspended for the window.',
     bg_color=MID, body_size=10)

card(s, 0.5,  3.38, 4.4, 1.62,
     'Maker / Taker Cost Split',
     'Limit orders (entries, normal exits) pay 0.03%. Market orders (stops, forced exits) pay 0.07%. Reflects real exchange fee structures.',
     bg_color=MID, body_size=10)

card(s, 5.1,  3.38, 4.4, 1.62,
     'Daily Borrow Fee',
     'Every short position pays an annual borrow rate pro-rated daily. On a 10-day hold, that\'s a real cost that compounds across thousands of trades.',
     bg_color=DARK, body_size=10)

# ── SLIDE 15 — Step 5 Section (dark hero) ───────────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
rect(s, 0.0, 0.0, 0.22, 5.625, GOLD)

tf = txb(s, 0.5, 1.1, 9.0, 0.42)
para(tf, 'THE VERIFICATION', size=11, bold=True, color=GOLD, first=True)

tf = txb(s, 0.5, 1.6, 9.0, 1.3)
para(tf, 'Proving It\'s Not Luck', font='Georgia', size=48, bold=True, color=WHITE, first=True)

tf = txb(s, 0.5, 3.05, 9.0, 0.52)
para(tf, 'Six independent tests. All must pass for the results to be trusted.', size=18, color=LBLUE, first=True)

tf = txb(s, 0.5, 3.7, 9.0, 0.7)
para(tf, 'A backtest can always be made to look good with enough parameter tuning. These tests separate real edges from overfitted noise.', size=13, italic=True, color=MUTED, first=True)

# ── SLIDE 16 — 6 Tests (light, 2×3 grid like agenda) ────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, LBGD)
tag(s, 'STEP 05 — VERIFICATION')
slide_title(s, 'Six Ways to Prove the Strategy Is Real')

tests = [
    ('01', 'Exit Z Grid',      '7 exit thresholds. If it only works at one specific value, it\'s fragile. If it works across all 7, the edge is real.'),
    ('02', 'Sniper Grid',      '84 parameter combos. Map the full (entry × exit × stop) space. A large "green zone" means robustness.'),
    ('03', 'Bootstrap',        'Shuffle all 10,000 trades randomly. In 95%+ of shuffles, the strategy must still be profitable.'),
    ('04', 'Dashboard',        'Equity curve, drawdown, win/loss histogram, rolling Sharpe, per-pair breakdown — all on one slide.'),
    ('05', 'Walk-Forward',     '6-month windows across 5 years. Did it work in 2020? 2022? 2023? Consistency beats a single lucky year.'),
    ('06', 'Trade Today?',     'GO / CAUTION / STOP per pair right now. Combines VIX, HMM, correlation, and MC confidence into a live signal.'),
]
positions = [(0.5,1.38),(5.18,1.38),(0.5,2.66),(5.18,2.66),(0.5,3.94),(5.18,3.94)]
for (num, title, desc), (lx, ty) in zip(tests, positions):
    rect(s, lx, ty, 4.4, 1.1, DARK)
    tf = txb(s, lx + 0.14, ty + 0.12, 0.55, 0.8)
    para(tf, num, font='Georgia', size=22, bold=True, color=GOLD, first=True)
    tf2 = txb(s, lx + 0.76, ty + 0.12, 3.46, 0.34)
    para(tf2, title, font='Georgia', size=12, bold=True, color=WHITE, first=True)
    tf3 = txb(s, lx + 0.76, ty + 0.50, 3.46, 0.52)
    para(tf3, desc, size=9.5, color=LBLUE, first=True)

# ── SLIDE 17 — How Many Backtests (dark, stats) ──────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
tag(s, 'STEP 05 — SCALE', dark_slide=True)
rect(s, 0.5, 0.52, 0.07, 0.72, GOLD)
tf = txb(s, 0.68, 0.52, 9.0, 0.72)
para(tf, 'How Many Times Does Backtesting Actually Run?', font='Georgia', size=26, bold=True, color=WHITE, first=True)

stat_block(s, 0.5,  1.56, '14',       'Distinct backtesting contexts across all steps')
stat_block(s, 2.9,  1.56, '50,000+',  'Individual simulations total (end-to-end)')
stat_block(s, 5.5,  1.56, '10,000',   'Monte Carlo paths per pair (Step 3d)')
stat_block(s, 8.0,  1.56, '84',       'Parameter combos per grid search')

tf = txb(s, 0.5, 2.9, 9.0, 2.4)
para(tf, 'The 14 Contexts Explained', font='Georgia', size=12, bold=True, color=GOLD, first=True)
rows2 = [
    '3a HMM comparison  ·  3d Monte Carlo (10,000 paths/pair)  ·  3f Grid train→test  ·  3g Regime-split grid',
    '3h Z-density scan  ·  3i R:R ratio scan  ·  3j Walk-forward (15 windows × 45 combos × N pairs)',
    '4a Full production backtest  ·  4b Strict/sniper  ·  4c Sniper 84-combo grid',
    '5a Exit-Z 7 values  ·  5b Sniper grid  ·  5c Bootstrap 10,000 shuffles  ·  5e Walk-forward stress test',
]
for row in rows2:
    para(tf, row, size=10, color=LBLUE, space_before=3)

tf2 = txb(s, 0.5, 5.22, 9.0, 0.28)
para(tf2, 'Step 3j alone generates thousands of mini-backtests — it accounts for the majority of all computation time in the pipeline.',
     size=10, italic=True, color=MUTED, first=True)

# ── SLIDE 18 — Conclusion (dark, hero) ──────────────────────────────────────
s = prs.slides.add_slide(BLK)
bg(s, NAVY)
rect(s, 0.0, 0.0, 0.22, 5.625, GOLD)
rect(s, 0.0, 4.85, 10.0, 0.775, DARK)

tf = txb(s, 0.42, 0.75, 9.2, 0.4)
para(tf, 'THE COMPLETE PIPELINE', size=11, bold=True, color=GOLD, first=True)

tf = txb(s, 0.42, 1.25, 9.0, 1.5)
para(tf, 'From 324 Pairs to a', font='Georgia', size=36, bold=True, color=WHITE, first=True)
para(tf, 'Live Trading Signal', font='Georgia', size=36, bold=True, color=GOLD)

flow = [
    ('Step 2', '324 candidates  →  20–40 confirmed pairs with proven leashes'),
    ('Step 3', '10 daily safety filters reduce bad trades before they happen'),
    ('Step 4', '5 years of data, every 5 minutes, with real costs — and a positive result'),
    ('Step 5', '14 verification contexts. Bootstrap, stress tests, walk-forward. It holds up.'),
]
for i, (step, desc) in enumerate(flow):
    lx = 0.42 + i * 2.38
    tf = txb(s, lx, 3.0, 2.2, 0.36)
    para(tf, step, font='Georgia', size=11, bold=True, color=GOLD, first=True)
    tf2 = txb(s, lx, 3.4, 2.2, 1.1)
    para(tf2, desc, size=9.5, color=LBLUE, first=True)
    if i < 3:
        tf3 = txb(s, lx + 2.2, 3.1, 0.18, 0.5)
        para(tf3, '→', font='Georgia', size=18, bold=True, color=GOLD, first=True)

tf = txb(s, 0.42, 4.92, 9.2, 0.5)
para(tf, 'AFES — Statistical Arbitrage Engine  ·  Quantitative Finance Club  ·  2026', size=11, color=MUTED, first=True)

# ── Save ─────────────────────────────────────────────────────────────────────
out = '/Users/frotidaa/Desktop/AFES_Presentation.pptx'
prs.save(out)
print(f'Saved → {out}')
print(f'Slides: {len(prs.slides)}')
