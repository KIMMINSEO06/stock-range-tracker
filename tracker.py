import glob
import os
import re
import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# 👇 설정 영역: 여러 종목을 넣고 실행 옵션을 고르세요
# =========================================================
STOCK_LIST = ["NVIDIA", "SK하이닉스", "삼성전자", "테슬라", "VOO", "IVV", "QQQ", "알파벳A", "알파벳B", "네이버", "카카오", "SNDK", "apple", "MSFT", "meta", "아마존", "AMD", "현대차", "LG전자"]
LOG_DIR = "stock_range_logs"

# 평소 매일 돌릴 때는 False, 3개월 백테스트 표를 함께 보려면 True
RUN_BACKTEST_FIRST = False


def resolve_ticker(query: str) -> tuple[str, str] | None:
    """반환: (market: 'US' 또는 'KR', ticker: str)"""
    q = (query or "").strip()
    if not q:
        return None

    kr_map = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "LG에너지솔루션": "373220.KS",
        "삼성바이오로직스": "207940.KS",
        "현대차": "005380.KS",
        "기아": "000270.KS",
        "NAVER": "035420.KS",
        "네이버": "035420.KS",
        "카카오": "035720.KS",
    }
    us_map = {
        "nvidia": "NVDA",
        "엔비디아": "NVDA",
        "apple": "AAPL",
        "애플": "AAPL",
        "tesla": "TSLA",
        "테슬라": "TSLA",
        "microsoft": "MSFT",
        "마이크로소프트": "MSFT",
        "amazon": "AMZN",
        "아마존": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "구글": "GOOGL",
        "meta": "META",
        "메타": "META",
        "amd": "AMD",
        "palantir": "PLTR",
    }

    if q in kr_map:
        return ("KR", kr_map[q])
    if q.upper().endswith(".KS") or q.upper().endswith(".KQ"):
        return ("KR", q.upper())
    if q.isdigit() and len(q) == 6:
        for sfx in (".KS", ".KQ"):
            try:
                test = yf.download(f"{q}{sfx}", period="5d", interval="1d", auto_adjust=True, progress=False)
                if test is not None and not test.empty:
                    return ("KR", f"{q}{sfx}")
            except Exception:
                continue

    low = q.lower()
    if low in us_map:
        return ("US", us_map[low])

    try:
        search = yf.Search(q, max_results=6)
        quotes = getattr(search, "quotes", []) or []
        for item in quotes:
            sym = str(item.get("symbol") or "").strip().upper()
            if sym.endswith(".KS") or sym.endswith(".KQ"):
                return ("KR", sym)
        us_exch = {"NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "NASDAQ", "NYSE"}
        for item in quotes:
            sym = str(item.get("symbol") or "").strip().upper()
            exch = str(item.get("exchange") or "").strip().upper()
            qtype = str(item.get("quoteType") or "").strip().upper()
            if sym and "." not in sym and (exch in us_exch or qtype == "EQUITY"):
                return ("US", sym)
    except Exception:
        pass

    cleaned = q.upper()
    if cleaned.replace("-", "").replace(".", "").isalnum() and len(cleaned) <= 10:
        return ("US", cleaned)
    return None


def get_stock_log_path(log_dir: str, ticker: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", ticker).strip("_")
    return os.path.join(log_dir, f"{safe_name}_range_log.csv")


def get_bias_icon(low_pct: float, high_pct: float) -> str:
    if abs(high_pct) > abs(low_pct):
        return "🔴"
    if abs(low_pct) > abs(high_pct):
        return "🟢"
    return "⚪"


def backtest_3m_single(query: str) -> dict:
    try:
        resolved = resolve_ticker(query)
        if not resolved:
            return {"입력": query, "티커": "-", "1일적중률": "실패", "1일오차": "-", "2일적중률": "실패", "2일오차": "-"}

        _, ticker = resolved
        raw = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return {"입력": query, "티커": ticker, "1일적중률": "데이터없음", "1일오차": "-", "2일적중률": "데이터없음", "2일오차": "-"}

        closes = raw["Close"].squeeze().dropna()
        if len(closes) < 30:
            return {"입력": query, "티커": ticker, "1일적중률": "표본부족", "1일오차": "-", "2일적중률": "표본부족", "2일오차": "-"}

        df = pd.DataFrame({"close": closes})
        df["chg_1d"] = df["close"].pct_change() * 100.0

        eval_start_idx = max(21, len(df) - 63)
        eval_end_idx = len(df) - 3

        records = []
        for i in range(eval_start_idx, eval_end_idx + 1):
            win_pct = df["chg_1d"].iloc[i - 19 : i + 1].dropna()
            if len(win_pct) < 15:
                continue
            mean_1d = float(win_pct.mean())
            std_1d = float(win_pct.std())
            if np.isnan(std_1d) or std_1d == 0:
                continue

            px_t = float(df["close"].iloc[i])
            px_t1 = float(df["close"].iloc[i + 1])
            px_t2 = float(df["close"].iloc[i + 2])

            act_1d_pct = (px_t1 - px_t) / px_t * 100.0
            act_2d_pct = (px_t2 - px_t) / px_t * 100.0

            d1_hit = ((mean_1d - std_1d) <= act_1d_pct <= (mean_1d + std_1d))
            d1_err = abs(act_1d_pct - mean_1d)

            d2_center = mean_1d * 2.0
            d2_std = std_1d * np.sqrt(2.0)
            d2_hit = ((d2_center - d2_std) <= act_2d_pct <= (d2_center + d2_std))
            d2_err = abs(act_2d_pct - d2_center)

            records.append({"d1_hit": d1_hit, "d1_err": d1_err, "d2_hit": d2_hit, "d2_err": d2_err})

        if not records:
            return {"입력": query, "티커": ticker, "1일적중률": "없음", "1일오차": "-", "2일적중률": "없음", "2일오차": "-"}

        res_df = pd.DataFrame(records)
        return {
            "입력": query,
            "티커": ticker,
            "1일적중률": f"{float(res_df['d1_hit'].mean()) * 100:.1f}%",
            "1일오차": f"{float(res_df['d1_err'].mean()):.2f}%p",
            "2일적중률": f"{float(res_df['d2_hit'].mean()) * 100:.1f}%",
            "2일오차": f"{float(res_df['d2_err'].mean()):.2f}%p",
        }
    except Exception as e:
        return {"입력": query, "티커": "-", "1일적중률": f"에러({type(e).__name__})", "1일오차": "-", "2일적중률": "-", "2일오차": "-"}


def run_backtest_summary(stock_list: list[str]):
    rows = [backtest_3m_single(item) for item in stock_list]
    print("=" * 78)
    print("🧪 최근 3개월 1일·2일 백테스트 요약")
    print("=" * 78)
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 78 + "\n")


def update_single_stock_log(log_path: str, ticker: str, closes_series: pd.Series) -> tuple[float, pd.DataFrame]:
    cols = [
        "ticker", "forecast_from_date", "base_close",
        "mean_1d", "std_1d", "used_k",
        "d1_low_pct", "d1_high_pct", "d1_badge",
        "actual_1d_date", "actual_1d_close", "actual_1d_pct",
        "in_range_1d", "abs_err_pct"
    ]
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
        except Exception:
            log_df = pd.DataFrame(columns=cols)
    else:
        log_df = pd.DataFrame(columns=cols)

    dates_str = [d.strftime("%Y-%m-%d") for d in closes_series.index]
    close_map = dict(zip(dates_str, [float(v) for v in closes_series.values]))

    for idx, row in log_df.iterrows():
        if pd.isna(row.get("actual_1d_pct")):
            f_date = str(row.get("forecast_from_date", ""))
            if f_date in close_map:
                later_dates = [d for d in dates_str if d > f_date]
                if later_dates:
                    next_d = later_dates[0]
                    prev_px = float(row["base_close"])
                    next_px = close_map[next_d]
                    if prev_px > 0:
                        act_pct = (next_px - prev_px) / prev_px * 100.0
                        in_rng = (float(row["d1_low_pct"]) <= act_pct <= float(row["d1_high_pct"]))
                        err_pct = abs(act_pct - float(row["mean_1d"]))
                        log_df.at[idx, "actual_1d_date"] = next_d
                        log_df.at[idx, "actual_1d_close"] = next_px
                        log_df.at[idx, "actual_1d_pct"] = round(act_pct, 4)
                        log_df.at[idx, "in_range_1d"] = bool(in_rng)
                        log_df.at[idx, "abs_err_pct"] = round(err_pct, 4)

    verified = log_df[log_df["actual_1d_pct"].notna()] if not log_df.empty else pd.DataFrame()
    adj_k = 1.0
    if len(verified) >= 5:
        hit_rate = float(verified["in_range_1d"].astype(bool).mean())
        diff_hit = 0.682 - hit_rate
        adj_k = float(np.clip(1.0 + diff_hit * 0.6, 0.75, 1.50))

    return adj_k, log_df


def process_one_stock_safely(query: str, log_dir: str = LOG_DIR) -> dict:
    try:
        resolved = resolve_ticker(query)
        if not resolved:
            return {"입력": query, "티커": "-", "상태": "❌ 매핑실패", "기준일": "-", "1일예상": "-", "2일예상": "-", "신호(1일/2일)": "-"}

        _, ticker = resolved
        log_path = get_stock_log_path(log_dir, ticker)

        raw = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return {"입력": query, "티커": ticker, "상태": "❌ 데이터없음", "기준일": "-", "1일예상": "-", "2일예상": "-", "신호(1일/2일)": "-"}

        closes = raw["Close"].squeeze().dropna()
        if len(closes) < 5:
            return {"입력": query, "티커": ticker, "상태": "❌ 데이터부족", "기준일": "-", "1일예상": "-", "2일예상": "-", "신호(1일/2일)": "-"}

        df = pd.DataFrame({"close": closes})
        df["chg_pct"] = df["close"].pct_change() * 100.0
        df_20d = df.tail(20).copy()

        valid_pct = df_20d["chg_pct"].dropna()
        mean_1d = float(valid_pct.mean())
        std_1d = float(valid_pct.std())
        if np.isnan(std_1d):
            std_1d = 0.0

        latest_px = float(df_20d["close"].iloc[-1])
        last_date = df_20d.index[-1].strftime("%Y-%m-%d")

        adj_k, log_df = update_single_stock_log(log_path, ticker, closes)
        eff_std_1d = std_1d * adj_k

        d1_low_pct = mean_1d - eff_std_1d
        d1_high_pct = mean_1d + eff_std_1d

        d2_mean_pct = mean_1d * 2.0
        d2_std_pct = eff_std_1d * np.sqrt(2.0)
        d2_low_pct = d2_mean_pct - d2_std_pct
        d2_high_pct = d2_mean_pct + d2_std_pct

        i1 = get_bias_icon(d1_low_pct, d1_high_pct)
        i2 = get_bias_icon(d2_low_pct, d2_high_pct)

        if not log_df.empty and "forecast_from_date" in log_df.columns:
            log_df = log_df[log_df["forecast_from_date"].astype(str) != last_date]

        new_row = {
            "ticker": ticker,
            "forecast_from_date": last_date,
            "base_close": round(latest_px, 4),
            "mean_1d": round(mean_1d, 4),
            "std_1d": round(std_1d, 4),
            "used_k": round(adj_k, 3),
            "d1_low_pct": round(d1_low_pct, 2),
            "d1_high_pct": round(d1_high_pct, 2),
            "d1_badge": f"{i1} {'상승폭' if i1 == '🔴' else '하락폭' if i1 == '🟢' else '동일'}",
            "actual_1d_date": np.nan,
            "actual_1d_close": np.nan,
            "actual_1d_pct": np.nan,
            "in_range_1d": np.nan,
            "abs_err_pct": np.nan,
        }
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
        log_df.to_csv(log_path, index=False)

        return {
            "입력": query,
            "티커": ticker,
            "상태": "✅ 완료",
            "기준일": last_date,
            "1일예상": f"{d1_low_pct:+.1f}%~{d1_high_pct:+.1f}%",
            "2일예상": f"{d2_low_pct:+.1f}%~{d2_high_pct:+.1f}%",
            "신호(1일/2일)": f"{i1} / {i2}",
        }
    except Exception as e:
        return {
            "입력": query,
            "티커": "-",
            "상태": f"⚠️ 에러({type(e).__name__})",
            "기준일": "-",
            "1일예상": "-",
            "2일예상": "-",
            "신호(1일/2일)": "-",
        }


def print_cumulative_summary(log_dir: str = LOG_DIR):
    pattern = os.path.join(log_dir, "*_range_log.csv")
    files = sorted(glob.glob(pattern))

    print("\n" + "=" * 78)
    print("📊 누적 로그 성적 요약 (전체 종목)")
    print("=" * 78)
    if not files:
        print("누적된 로그 파일이 아직 없습니다.")
        print("=" * 78)
        return

    rows = []
    for path in files:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            ticker = str(df["ticker"].dropna().iloc[-1]) if "ticker" in df.columns and not df["ticker"].dropna().empty else os.path.basename(path)
            verified = df[df["actual_1d_pct"].notna()] if "actual_1d_pct" in df.columns else pd.DataFrame()
            v_cnt = len(verified)
            if v_cnt > 0:
                hit_str = f"{float(verified['in_range_1d'].astype(bool).mean()) * 100.0:.1f}%"
                err_str = f"{float(verified['abs_err_pct'].mean()):.2f}%p"
            else:
                hit_str = "-"
                err_str = "-"
            k_val = float(df["used_k"].dropna().iloc[-1]) if "used_k" in df.columns and not df["used_k"].dropna().empty else 1.0
            rows.append({
                "종목": ticker,
                "기록수": len(df),
                "검증수": v_cnt,
                "누적적중률": hit_str,
                "평균오차": err_str,
                "배수(k)": f"{k_val:.2f}x",
            })
        except Exception as e:
            rows.append({
                "종목": os.path.basename(path),
                "기록수": "-",
                "검증수": "-",
                "누적적중률": f"읽기에러({type(e).__name__})",
                "평균오차": "-",
                "배수(k)": "-",
            })

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 78)


if RUN_BACKTEST_FIRST:
    run_backtest_summary(STOCK_LIST)

run_rows = [process_one_stock_safely(item, LOG_DIR) for item in STOCK_LIST]
print("=" * 78)
print("📌 오늘 배치 실행 요약")
print("=" * 78)
print(pd.DataFrame(run_rows).to_string(index=False))
print("=" * 78)

print_cumulative_summary(LOG_DIR)
