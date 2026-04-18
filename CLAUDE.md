# Polymarket Weather Arbitrage Dashboard

## 專案概述

Python/FastAPI 後端 + 單頁 HTML 前端，部署在 Railway。
掃描 Polymarket 天氣相關預測市場，用 NWS/MetNo/WU 氣象數據計算套利機會。

- **GitHub**: https://github.com/Azstarrr/weather.git (master branch)
- **本地路徑**: `C:\Users\yuga2\Desktop\claude\polymarket-weather\`
- **語言偏好**: 用戶溝通使用中文

---

## 架構

```
server.py          FastAPI 主服務，port=$PORT, host="0.0.0.0"
comparator.py      掃描引擎：compare_all_markets(), Kalman filter, 套利計算
consensus.py       氣象共識層（urllib, ssl.CERT_NONE）
fetcher_nws.py     NWS 美國氣象 API（urllib, ssl.CERT_NONE）
fetcher_metno.py   Met.no 國際氣象 API（urllib, ssl.CERT_NONE）
_http.py           共用 aiohttp.ClientSession（timeout 30s/10s）
static/index.html  單頁前端，繁體中文 UI
railway.toml       Railway 部署設定
requirements.txt   無 requests（已移除）
```

---

## 關鍵設計決策

### 非阻塞掃描架構
- `POST /api/refresh` → 立即返回，後台 `asyncio.create_task(_run_scan_task())`
- `GET /api/opportunities` → 返回快取 + `scanning: true/false` flag
- 前端輪詢：每 3 秒 GET，最多等 180 秒

### SSL（非美國地點）
所有 urllib 呼叫使用 `ssl.CERT_NONE`：
```python
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE
```
原因：Windows Railway 容器缺少 Comodo/Sectigo 中間憑證

### 市場掃描限制
- 上限：400 個市場（`[:400]`）
- 分批處理：`_CHUNK_SIZE = 100`，每批間 50ms yield
- 共識計算：僅對 top-10 機會執行（節省 15-20s）

### 時區修正
UTC+8 台灣時間 4/17 查看時，美國 4/16 市場不應被標記為過期：
```python
now_utc = datetime.now(timezone.utc)
today_utc = now_utc.date()
days_ahead = (target_date - today_utc).days
if days_ahead < 0:
    local_offset = _get_utc_offset(location, 0.0, dt=now_utc)
    local_date = (now_utc + timedelta(hours=local_offset)).date()
    days_ahead = (target_date - local_date).days
```

### CLOB 定價（重要陷阱）
`fetch_clob_book` 是 `async def`，**必須用 `await`**：
```python
book = await fetch_clob_book(token_id_yes)  # 不能漏掉 await
```
漏掉 await 會靜默返回 coroutine 物件，導致所有市場用估算價差。

---

## 前端 Filter Tabs

```
全部 | 🎯 套利機會 | ⚡ Kalman重置 | 🌊 高滑點 | 📈 高Kelly
```

- `opp`: `r.is_opportunity === true`
- `regime`: Kalman 狀態重置
- `slippage`: 高滑點警告
- `high-kelly`: Kelly 值 > 閾值

---

## 已修復的 Bug（勿重蹈）

| Bug | 原因 | 修復 |
|-----|------|------|
| `ModuleNotFoundError: requests` | requirements.txt 移除 requests 但程式還在用 | 改用 urllib |
| Railway Mock 數據 | aiohttp timeout=15s 太短 | 改 30s/10s |
| 刷新按鈕無回應 | 30s HTTP timeout vs 60-120s 掃描 | 背景任務 + 前端輪詢 |
| SSL CERTIFICATE_VERIFY_FAILED | Windows CA 缺少憑證 | ssl.CERT_NONE |
| CLOB 定價全部錯誤 | 漏掉 await fetch_clob_book | 補上 await |
| 時區提前過期 | date.today() 用 UTC | 改用城市本地時間 |

---

## 待驗證（Railway 上）

- [ ] 不再出現 Mock 數據警告
- [ ] 刷新按鈕正常（顯示倒數 → 完成）
- [ ] 套利機會 filter tab 計數正確
- [ ] 台灣時間看美國前一天市場仍顯示
