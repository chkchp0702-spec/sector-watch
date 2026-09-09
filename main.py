"""
sector-watch (단일 파일 버전) — 매주 월요일 섹터 로테이션 & 관심종목 리포트
파일 4개만 있으면 됩니다: main.py, config.yaml, requirements.txt, .github/workflows/weekly.yml
"""
import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore", message="Glyph")

# ======================================================================
# DATA
# ======================================================================
# OHLCV 수집: pykrx 우선, 실패 시 yfinance(.KS) 폴백.

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _from_pykrx(code: str, start: str, end: str) -> pd.DataFrame | None:
    from pykrx import stock
    for fn in (stock.get_etf_ohlcv_by_date, stock.get_market_ohlcv_by_date):
        try:
            df = fn(start, end, code)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low",
                                "종가": "Close", "거래량": "Volume"})
        if not set(COLS).issubset(df.columns):
            continue
        df.index = pd.to_datetime(df.index)
        df = df[COLS].astype(float)
        return df[df["Close"] > 0]
    return None


def _from_yf(code: str, start: str) -> pd.DataFrame | None:
    import yfinance as yf
    ticker = code if "." in code else f"{code}.KS"
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[COLS].dropna().astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_ohlcv(code: str, lookback_days: int = 300, source: str = "pykrx") -> pd.DataFrame | None:
    end = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.7))  # 휴장일 여유
    s8, e8 = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    df = None
    if source == "pykrx":
        try:
            df = _from_pykrx(code, s8, e8)
        except Exception as e:
            print(f"[pykrx 실패] {code}: {e}")
    if df is None or len(df) < 130:
        try:
            df = _from_yf(code, start.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"[yfinance 실패] {code}: {e}")
    if df is None or len(df) < 130:
        print(f"[데이터 부족] {code}")
        return None
    return df.tail(lookback_days)


def fetch_many(codes: dict[str, str], lookback_days: int, source: str) -> dict[str, pd.DataFrame]:
    out = {}
    for name, code in codes.items():
        df = fetch_ohlcv(code, lookback_days, source)
        if df is not None:
            out[name] = df
    return out


# ─────────────────────────── KRX 업종지수 ───────────────────────────
def fetch_index_ohlcv(code: str, lookback_days: int = 300) -> pd.DataFrame | None:
    from pykrx import stock
    end = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.7))
    try:
        df = stock.get_index_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code)
    except Exception as e:
        print(f"[지수 실패] {code}: {e}")
        return None
    if df is None or df.empty:
        print(f"[지수 없음] {code}")
        return None
    df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low",
                            "종가": "Close", "거래량": "Volume"})
    df.index = pd.to_datetime(df.index)
    df = df[COLS].astype(float)
    return df[df["Close"] > 0].tail(lookback_days)


def fetch_sectors(cfg: dict) -> dict[str, pd.DataFrame]:
    """sector_mode 에 따라 업종지수 또는 ETF 로 섹터 시계열 수집."""
    if cfg.get("sector_mode", "index") == "index":
        out = {}
        for name, code in cfg["sectors_index"].items():
            df = fetch_index_ohlcv(code, cfg["lookback_days"])
            if df is not None:
                out[name] = df
        if len(out) >= 3:
            return out
        print("[경고] 업종지수 수집 실패 → ETF 모드로 폴백")
    return fetch_many(cfg["sectors_etf"], cfg["lookback_days"], "pykrx")


def index_constituents(code: str) -> list[str]:
    """업종지수 구성종목 코드 목록."""
    from pykrx import stock
    try:
        return list(stock.get_index_portfolio_deposit_file(code))
    except Exception as e:
        print(f"[구성종목 실패] {code}: {e}")
        return []


def market_caps(tickers: list[str]) -> pd.Series:
    """종목코드 → 시가총액. 최근 영업일 기준."""
    from pykrx import stock
    for back in range(0, 7):
        d = (datetime.today() - timedelta(days=back)).strftime("%Y%m%d")
        try:
            cap = stock.get_market_cap_by_ticker(d, market="ALL")
            if cap is not None and not cap.empty and cap["시가총액"].sum() > 0:
                return cap["시가총액"].reindex(tickers).dropna()
        except Exception:
            continue
    return pd.Series(dtype=float)


def ticker_name(code: str) -> str:
    from pykrx import stock
    try:
        return stock.get_market_ticker_name(code) or code
    except Exception:
        return code

# ======================================================================
# SECTOR
# ======================================================================
# 섹터별 기간 누적수익률 맵 · 순위 · 순위변동 시그널.

PERIODS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "12M": 252}
RANK_HISTORY = Path("data/rank_history.csv")


def period_returns(ohlcv: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, df in ohlcv.items():
        c = df["Close"]
        rows[name] = {p: (c.iloc[-1] / c.iloc[-n - 1] - 1) * 100 if len(c) > n else float("nan")
                      for p, n in PERIODS.items()}
    return pd.DataFrame(rows).T[list(PERIODS)]


def rank_table(ret: pd.DataFrame) -> pd.DataFrame:
    return ret.rank(ascending=False, method="min").astype("Int64")


def load_prev_ranks() -> pd.DataFrame | None:
    if not RANK_HISTORY.exists():
        return None
    h = pd.read_csv(RANK_HISTORY)
    last_date = h["date"].max()
    prev = h[h["date"] == last_date].set_index("sector")[list(PERIODS)]
    return prev, last_date


def save_ranks(rank: pd.DataFrame, date: str):
    df = rank.copy()
    df.insert(0, "sector", df.index)
    df.insert(0, "date", date)
    RANK_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    if RANK_HISTORY.exists():
        old = pd.read_csv(RANK_HISTORY)
        old = old[old["date"] != date]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RANK_HISTORY, index=False)


def detect_signals(rank: pd.DataFrame, prev: pd.DataFrame | None, cfg: dict) -> list[dict]:
    """
    반환: [{sector, level('경고'|'주목'|'지속'), tag, detail}]
    - 주도 약화 경고: 12M/6M 상위권인데 1M 또는 1W가 하위 절반으로 밀림
    - 신규 주도 후보: 12M/6M 하위권인데 1W 또는 1M 상위권 진입
    - 순위 급변: 직전 리포트 대비 1W/1M/3M 순위가 임계값 이상 변동
    - 주도 지속: 모든 기간 상위권
    """
    n = len(rank)
    top = cfg["top_n"]
    half = n / 2
    out = []

    for s, r in rank.iterrows():
        r = r.astype(float)
        long_top = min(r["12M"], r["6M"]) <= top
        long_bottom = max(r["12M"], r["6M"]) > n - top
        short_top = min(r["1W"], r["1M"]) <= top
        short_weak = max(r["1W"], r["1M"]) > half

        if long_top and short_weak:
            out.append(dict(sector=s, level="경고", tag="주도 약화",
                            detail=f"12M {int(r['12M'])}위·6M {int(r['6M'])}위 → 1M {int(r['1M'])}위·1W {int(r['1W'])}위"))
        elif long_bottom and short_top:
            out.append(dict(sector=s, level="주목", tag="신규 주도 후보",
                            detail=f"12M {int(r['12M'])}위·6M {int(r['6M'])}위 → 1M {int(r['1M'])}위·1W {int(r['1W'])}위"))
        elif all(r[p] <= top for p in PERIODS):
            out.append(dict(sector=s, level="지속", tag="주도 지속", detail="전 기간 상위권"))

        if prev is not None and s in prev.index:
            for p, key in (("1W", "rank_jump_1w"), ("1M", "rank_jump_1m"), ("3M", "rank_jump_3m")):
                try:
                    d = int(prev.loc[s, p]) - int(r[p])  # +면 순위 상승
                except (ValueError, TypeError):
                    continue
                if abs(d) >= cfg[key]:
                    lvl, arrow = ("주목", "▲") if d > 0 else ("경고", "▼")
                    out.append(dict(sector=s, level=lvl, tag=f"{p} 순위 급변",
                                    detail=f"{int(prev.loc[s, p])}위 → {int(r[p])}위 ({arrow}{abs(d)})"))

    order = {"경고": 0, "주목": 1, "지속": 2}
    return sorted(out, key=lambda x: (order[x["level"]], x["sector"]))

# ======================================================================
# TECHNICAL
# ======================================================================
# 기술적 지표: 이평선 크로스 · 거래량 · 120거래일 매물벽 · Cup with Handle.

MA_WINDOWS = (5, 20, 60, 120)


# ─────────────────────────── 이동평균 / 크로스 ───────────────────────────
def ma_status(df: pd.DataFrame, lookback: int) -> dict:
    c = df["Close"]
    ma = {w: c.rolling(w).mean() for w in MA_WINDOWS}
    price = c.iloc[-1]

    above = {f"MA{w}": bool(price > ma[w].iloc[-1]) for w in MA_WINDOWS}
    vals = [ma[w].iloc[-1] for w in MA_WINDOWS]
    if price > vals[0] > vals[1] > vals[2] > vals[3]:
        align = "정배열"
    elif price < vals[0] < vals[1] < vals[2] < vals[3]:
        align = "역배열"
    else:
        align = "혼조"

    crosses = []
    pairs = [(5, 20), (20, 60), (60, 120)]
    for s, l in pairs:
        diff = (ma[s] - ma[l]).dropna()
        if len(diff) < lookback + 1:
            continue
        recent = diff.iloc[-(lookback + 1):]
        sign = np.sign(recent)
        for i in range(1, len(sign)):
            if sign.iloc[i - 1] <= 0 < sign.iloc[i]:
                crosses.append(f"골든 {s}/{l} ({recent.index[i].date()})")
            elif sign.iloc[i - 1] >= 0 > sign.iloc[i]:
                crosses.append(f"데드 {s}/{l} ({recent.index[i].date()})")
    # 현재가 vs 이평선 돌파/이탈
    for w in MA_WINDOWS:
        d = (c - ma[w]).dropna().iloc[-(lookback + 1):]
        sg = np.sign(d)
        if len(sg) > 1:
            if sg.iloc[-1] > 0 and (sg.iloc[:-1] <= 0).any():
                crosses.append(f"MA{w} 상향돌파")
            elif sg.iloc[-1] < 0 and (sg.iloc[:-1] >= 0).any():
                crosses.append(f"MA{w} 하향이탈")

    return dict(price=price, ma={f"MA{w}": round(ma[w].iloc[-1], 1) for w in MA_WINDOWS},
                above=above, align=align, crosses=crosses)


# ─────────────────────────── 거래량 ───────────────────────────
def volume_status(df: pd.DataFrame, surge: float, dry: float) -> dict:
    v = df["Volume"]
    r5_20 = v.tail(5).mean() / max(v.tail(20).mean(), 1)
    r20_60 = v.tail(20).mean() / max(v.tail(60).mean(), 1)
    if r5_20 >= surge:
        label = "급증"
    elif r5_20 <= dry:
        label = "감소(마름)"
    else:
        label = "보합"
    return dict(ratio_5_20=round(r5_20, 2), ratio_20_60=round(r20_60, 2), label=label,
                last_vs_20=round(v.iloc[-1] / max(v.tail(20).mean(), 1), 2))


# ─────────────────────────── 120거래일 매물벽 ───────────────────────────
def supply_wall(df: pd.DataFrame, bins: int = 24, window: int = 120) -> dict:
    d = df.tail(window)
    price = d["Close"].iloc[-1]
    lo, hi = d["Low"].min(), d["High"].max()
    edges = np.linspace(lo, hi, bins + 1)
    # 각 봉의 거래량을 (고가~저가) 구간에 균등 분배
    vol = np.zeros(bins)
    for _, row in d.iterrows():
        idx = np.where((edges[:-1] < row["High"]) & (edges[1:] > row["Low"]))[0]
        if len(idx):
            vol[idx] += row["Volume"] / len(idx)
    centers = (edges[:-1] + edges[1:]) / 2
    total = vol.sum() or 1
    share = vol / total

    above = [(c, s) for c, s in zip(centers, share) if c > price]
    below = [(c, s) for c, s in zip(centers, share) if c <= price]
    # 상단 매물벽: 현재가 위 구간 중 비중이 평균 이상인 첫 구간
    avg = 1 / bins
    wall = next(((c, s) for c, s in above if s >= avg * 1.3), None)
    heavy_above = sum(s for _, s in above)  # 현재가 위 총 매물 비중
    poc = centers[int(np.argmax(vol))]      # 최대 매물대

    return dict(
        wall_price=round(wall[0], 0) if wall else None,
        wall_dist_pct=round((wall[0] / price - 1) * 100, 1) if wall else None,
        wall_share_pct=round(wall[1] * 100, 1) if wall else None,
        overhead_share_pct=round(heavy_above * 100, 1),
        poc=round(poc, 0), poc_dist_pct=round((poc / price - 1) * 100, 1),
        position="매물대 상단(가벼움)" if heavy_above < 0.2 else
                 "매물대 중간" if heavy_above < 0.5 else "매물대 하단(무거움)",
    )


# ─────────────────────────── Cup with Handle ───────────────────────────
def cup_with_handle(df: pd.DataFrame) -> dict:
    """
    휴리스틱 (O'Neil 기준 완화판):
      - 좌측 고점: 최근 250~30일 구간의 최고가
      - 컵 깊이 12~50%, 컵 길이 ≥ 30거래일
      - 우측 회복: 저점 이후 고가가 좌측 고점의 85% 이상
      - 핸들: 마지막 5~25거래일, 되돌림 ≤ 15%, 컵 중간값 위, 거래량 감소
    반환: detected, score(0~100), pivot(핸들 고점=매수 포인트), 상세
    """
    if len(df) < 130:
        return dict(detected=False, score=0, note="데이터 부족")
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    n = len(df)
    win = df.iloc[max(0, n - 250): n - 30]
    if win.empty:
        return dict(detected=False, score=0)
    left_i = win["High"].idxmax()
    left_peak = h[left_i]
    after = df.loc[left_i:]
    bottom_i = after["Low"].idxmin()
    bottom = l[bottom_i]
    depth = 1 - bottom / left_peak
    cup_len = len(df.loc[left_i:bottom_i])
    right = df.loc[bottom_i:]
    right_peak_i = right["High"].idxmax()
    right_peak = h[right_peak_i]
    recovery = right_peak / left_peak

    # 핸들: 우측 고점 이후 구간
    handle = df.loc[right_peak_i:].iloc[1:]
    handle_len = len(handle)
    handle_dd = 1 - handle["Low"].min() / right_peak if handle_len else 0
    mid = (left_peak + bottom) / 2
    handle_above_mid = handle["Low"].min() > mid if handle_len else False
    handle_vol_dry = (handle["Volume"].mean() < right["Volume"].mean()) if handle_len else False

    checks = {
        "깊이 12~50%": 0.12 <= depth <= 0.50,
        "컵 길이 ≥30일": cup_len >= 30,
        "우측 회복 ≥85%": recovery >= 0.85,
        "핸들 5~25일": 5 <= handle_len <= 25,
        "핸들 되돌림 ≤15%": 0 < handle_dd <= 0.15,
        "핸들 컵 상단 절반": handle_above_mid,
        "핸들 거래량 감소": handle_vol_dry,
    }
    score = int(100 * sum(checks.values()) / len(checks))
    pivot = right_peak
    breakout = c.iloc[-1] > pivot * 1.0
    return dict(
        detected=score >= 70, score=score, checks=checks,
        left_peak=round(left_peak, 0), bottom=round(bottom, 0), depth_pct=round(depth * 100, 1),
        cup_len=cup_len, handle_len=handle_len, handle_dd_pct=round(handle_dd * 100, 1),
        pivot=round(pivot, 0), pivot_dist_pct=round((pivot / c.iloc[-1] - 1) * 100, 1),
        breakout=bool(breakout),
    )


def analyze(df: pd.DataFrame, cfg: dict) -> dict:
    return dict(
        ma=ma_status(df, cfg["cross_lookback"]),
        vol=volume_status(df, cfg["volume_surge_ratio"], cfg["volume_dry_ratio"]),
        wall=supply_wall(df, cfg["supply_wall_bins"]),
        cup=cup_with_handle(df),
    )

# ======================================================================
# WATCHLIST
# ======================================================================
# 관심종목 자동 편입/제외 — data/watchlist_auto.yaml 에 상태 저장.


STATE = Path("data/watchlist_auto.yaml")


def load_state() -> dict:
    if STATE.exists():
        return yaml.safe_load(STATE.read_text(encoding="utf-8")) or {"items": {}, "weak_streak": {}}
    return {"items": {}, "weak_streak": {}}


def save_state(state: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(yaml.safe_dump(state, allow_unicode=True, sort_keys=False), encoding="utf-8")


def pick_leaders(sector: str, index_code: str, cfg: dict, lookback: int) -> list[dict]:
    """섹터 구성종목 중 시총 상위 → 3M 수익률 상위 pick_top 개."""
    codes = index_constituents(index_code)
    if not codes:
        return []
    caps = market_caps(codes).sort_values(ascending=False).head(cfg["candidates_by_cap"])
    scored = []
    for code in caps.index:
        df = fetch_ohlcv(code, lookback, "pykrx")
        if df is None or len(df) < 64:
            continue
        c = df["Close"]
        r3m = (c.iloc[-1] / c.iloc[-64] - 1) * 100
        r1m = (c.iloc[-1] / c.iloc[-22] - 1) * 100
        scored.append(dict(code=code, name=ticker_name(code), cap=float(caps[code]),
                           r1m=round(r1m, 1), r3m=round(r3m, 1)))
    scored = [x for x in scored if x["r3m"] > 0]          # 하락 종목은 제외
    scored.sort(key=lambda x: x["r3m"], reverse=True)
    return scored[: cfg["pick_top"]]


def update(signals: list[dict], sectors_index: dict, cfg: dict, lookback: int, today: str) -> dict:
    """
    반환: {added:[...], removed:[...], review:[...], items:{code:{...}}}
    - 신규 주도 후보 / 주도 지속 섹터 → 구성 주도종목 편입
    - 주도 약화 섹터가 remove_after_weeks 연속 → 그 섹터 종목 제외
    """
    state = load_state()
    items, streak = state["items"], state["weak_streak"]
    added, removed, review = [], [], []

    strong = {s["sector"] for s in signals if s["tag"] in ("신규 주도 후보", "주도 지속")}
    weak = {s["sector"] for s in signals if s["tag"] == "주도 약화"}

    for sec in strong:
        if sec not in sectors_index:
            continue
        for p in pick_leaders(sec, sectors_index[sec], cfg, lookback):
            if p["code"] not in items:
                items[p["code"]] = dict(name=p["name"], sector=sec, added=today,
                                        reason=f"{sec} 주도종목 (3M {p['r3m']:+}%)")
                added.append(f"{p['name']}({p['code']}) — {sec}, 3M {p['r3m']:+}%")

    for sec in list(set(streak) | weak):
        streak[sec] = streak.get(sec, 0) + 1 if sec in weak else 0
        if streak[sec] == 0:
            streak.pop(sec, None)
    for sec, n in streak.items():
        victims = [c for c, v in items.items() if v["sector"] == sec]
        if n >= cfg["remove_after_weeks"]:
            for c in victims:
                removed.append(f"{items[c]['name']}({c}) — {sec} {n}주 연속 약화")
                items.pop(c)
        elif victims:
            review.append(f"{sec}: " + ", ".join(items[c]["name"] for c in victims) + f" (약화 {n}주차)")

    state["items"], state["weak_streak"] = items, streak
    save_state(state)
    return dict(added=added, removed=removed, review=review, items=items)

# ======================================================================
# REPORT
# ======================================================================
# Markdown 리포트 + 히트맵 PNG + 텔레그램 요약.


LEVEL_ICON = {"경고": "🔴", "주목": "🟢", "지속": "🔵"}


def _set_korean_font():
    for name in ("NanumGothic", "Malgun Gothic", "AppleGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def heatmap(ret: pd.DataFrame, rank: pd.DataFrame, path: Path):
    _set_korean_font()
    data = ret.loc[rank["1M"].astype(int).sort_values().index]  # 1M 순위순 정렬
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.5))
    vmax = float(data.abs().max().max()) or 1
    im = ax.imshow(data.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(data.columns)), data.columns)
    ax.set_yticks(range(len(data.index)), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            rk = rank.loc[data.index[i], data.columns[j]]
            ax.text(j, i, f"{val:+.1f}%\n({rk})", ha="center", va="center", fontsize=8)
    ax.set_title("섹터별 누적 수익률 (괄호: 순위, 1M 순위순)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _md_table(ret: pd.DataFrame, rank: pd.DataFrame, prev: pd.DataFrame | None) -> str:
    order = rank["1M"].astype(int).sort_values().index
    head = "| 섹터 | " + " | ".join(PERIODS) + " |\n|---|" + "---|" * len(PERIODS) + "\n"
    rows = []
    for s in order:
        cells = []
        for p in PERIODS:
            r, k = ret.loc[s, p], int(rank.loc[s, p])
            delta = ""
            if prev is not None and s in prev.index and pd.notna(prev.loc[s, p]):
                d = int(prev.loc[s, p]) - k
                delta = f" ▲{d}" if d > 0 else f" ▼{-d}" if d < 0 else ""
            cells.append(f"{r:+.1f}% ({k}위{delta})")
        rows.append(f"| **{s}** | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def _tech_block(name: str, code: str, t: dict) -> str:
    ma, vol, wall, cup = t["ma"], t["vol"], t["wall"], t["cup"]
    lines = [f"### {name} ({code}) — 현재가 {ma['price']:,.0f}"]
    lines.append(f"- **이평선**: {ma['align']} · " +
                 " / ".join(f"{k} {v:,.0f}{'▲' if ma['above'][k] else '▼'}" for k, v in ma["ma"].items()))
    lines.append("- **크로스(최근)**: " + (", ".join(ma["crosses"]) if ma["crosses"] else "없음"))
    lines.append(f"- **거래량**: {vol['label']} · 5/20일 {vol['ratio_5_20']}x · 20/60일 {vol['ratio_20_60']}x · 당일/20일 {vol['last_vs_20']}x")
    if wall["wall_price"]:
        lines.append(f"- **120일 매물벽**: {wall['wall_price']:,.0f} (+{wall['wall_dist_pct']}%, 비중 {wall['wall_share_pct']}%) · "
                     f"상단 총매물 {wall['overhead_share_pct']}% → {wall['position']} · 최대매물대 {wall['poc']:,.0f} ({wall['poc_dist_pct']:+}%)")
    else:
        lines.append(f"- **120일 매물벽**: 현재가 위 유의미한 매물대 없음 (상단 {wall['overhead_share_pct']}%) → {wall['position']}")
    if cup.get("score", 0) > 0:
        flag = "✅ 패턴 성립" if cup["detected"] else "패턴 미성립"
        bo = " · **피벗 돌파 중**" if cup.get("breakout") else ""
        lines.append(f"- **Cup&Handle**: {flag} (점수 {cup['score']}) · 깊이 {cup['depth_pct']}% · 컵 {cup['cup_len']}일 · "
                     f"핸들 {cup['handle_len']}일/되돌림 {cup['handle_dd_pct']}% · 피벗 {cup['pivot']:,.0f} ({cup['pivot_dist_pct']:+}%){bo}")
        missed = [k for k, v in cup["checks"].items() if not v]
        if missed:
            lines.append(f"  - 미충족: {', '.join(missed)}")
    return "\n".join(lines)


def build_markdown(date: str, mode: str, ret, rank, prev, prev_date, signals, tech: dict,
                   wl: dict, heatmap_rel: str) -> str:
    md = [f"# 섹터 로테이션 & 관심종목 리포트 — {date} ({mode})", ""]
    md.append(f"![heatmap]({heatmap_rel})\n")

    md.append("## 1. 시그널")
    if signals:
        for s in signals:
            md.append(f"- {LEVEL_ICON[s['level']]} **[{s['tag']}] {s['sector']}** — {s['detail']}")
    else:
        md.append("- 특이 시그널 없음")
    if prev_date:
        md.append(f"\n_순위변동 기준: {prev_date} 리포트 대비_")
    md.append("")

    md.append("## 2. 섹터 누적수익률 맵 (1M 순위순)")
    md.append(_md_table(ret, rank, prev))
    md.append("")

    md.append("## 3. 관심종목 자동 업데이트")
    if wl["added"]:
        md.append("**✅ 신규 편입**")
        md += [f"- {x}" for x in wl["added"]]
    if wl["removed"]:
        md.append("**❌ 제외**")
        md += [f"- {x}" for x in wl["removed"]]
    if wl["review"]:
        md.append("**⚠️ 제외 검토 (섹터 약화 진행 중)**")
        md += [f"- {x}" for x in wl["review"]]
    if not (wl["added"] or wl["removed"] or wl["review"]):
        md.append("변동 없음")
    if wl["items"]:
        md.append("\n**현재 자동 관심종목**")
        md.append("| 종목 | 코드 | 섹터 | 편입일 | 사유 |\n|---|---|---|---|---|")
        md += [f"| {v['name']} | {c} | {v['sector']} | {v['added']} | {v['reason']} |" for c, v in wl["items"].items()]
    md.append("")

    md.append("## 4. 기술적 지표 점검")
    for name, (code, t) in tech.items():
        md.append(_tech_block(name, code, t))
        md.append("")
    md.append("---\n_자동 생성. 투자 판단의 참고 자료이며 매매 권유가 아닙니다._")
    return "\n".join(md)


def telegram_summary(date: str, signals: list, tech: dict, wl: dict) -> str:
    lines = [f"📊 섹터 리포트 {date}"]
    for s in signals[:8]:
        lines.append(f"{LEVEL_ICON[s['level']]} [{s['tag']}] {s['sector']}: {s['detail']}")
    hot = [n for n, (_, t) in tech.items()
           if t["cup"].get("detected") or any(c.startswith("골든") for c in t["ma"]["crosses"])]
    if hot:
        lines.append("⭐ 기술적 주목: " + ", ".join(hot))
    if wl["added"]:
        lines.append("✅ 편입: " + " / ".join(wl["added"]))
    if wl["removed"]:
        lines.append("❌ 제외: " + " / ".join(wl["removed"]))
    return "\n".join(lines)


def send_telegram(text: str):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    import requests
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=15)
    except Exception as e:
        print(f"[telegram 실패] {e}")


# ======================================================================
# MAIN
# ======================================================================
def list_indices():
    """업종지수 코드 확인: LIST_INDICES=1 python main.py"""
    from pykrx import stock
    for mkt in ("KOSPI", "KOSDAQ"):
        print(f"\n=== {mkt} ===")
        for t in stock.get_index_ticker_list(market=mkt):
            print(t, stock.get_index_ticker_name(t))


def main():
    if os.getenv("LIST_INDICES"):
        return list_indices()
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    today = date.today()
    mode = os.getenv("MODE", "auto")
    if mode == "auto":
        mode = "monthly" if today.day <= 7 else "weekly"   # 매월 첫 월요일
    ds = today.isoformat()
    print(f"== {ds} / {mode} ==")

    # 1) 섹터 수익률 맵 & 시그널
    sec = fetch_sectors(cfg)
    if len(sec) < 3:
        raise SystemExit("섹터 데이터 수집 실패")
    ret = period_returns(sec)
    rank = rank_table(ret)
    prev_pack = load_prev_ranks()
    prev, prev_date = prev_pack if prev_pack else (None, None)
    signals = detect_signals(rank, prev, cfg["signals"])
    save_ranks(rank, ds)

    # 2) 관심종목 자동 편입/제외 (업종지수 모드에서만 구성종목 조회 가능)
    if cfg.get("sector_mode", "index") == "index":
        wl = update(signals, cfg["sectors_index"], cfg["auto_watchlist"],
                              cfg["lookback_days"], ds)
    else:
        wl = dict(added=[], removed=[], review=[], items={})

    # 3) 기술적 지표: 수동 관심종목 + 자동 편입 종목 (+ monthly: 전 섹터)
    codes = dict(cfg["watchlist"])
    for c, v in wl["items"].items():
        codes.setdefault(f"{v['name']} [{v['sector']}]", c)
    tech = {}
    for name, code in codes.items():
        df = fetch_ohlcv(code, cfg["lookback_days"], "pykrx")
        if df is not None:
            tech[name] = (code, analyze(df, cfg["signals"]))
    if mode == "monthly":
        for name, df in sec.items():
            tech[f"[섹터] {name}"] = ("-", analyze(df, cfg["signals"]))

    # 4) 출력
    out = Path("reports"); out.mkdir(exist_ok=True)
    png = out / f"heatmap_{ds}.png"
    heatmap(ret, rank, png)
    md = build_markdown(ds, mode, ret, rank, prev, prev_date, signals, tech, wl, png.name)
    (out / f"{ds}.md").write_text(md, encoding="utf-8")
    (out / "latest.md").write_text(md, encoding="utf-8")
    (out / "heatmap_latest.png").write_bytes(png.read_bytes())
    ret.round(2).to_csv(out / f"returns_{ds}.csv", encoding="utf-8-sig")

    send_telegram(telegram_summary(ds, signals, tech, wl))
    print(md[:2000])


if __name__ == "__main__":
    main()
