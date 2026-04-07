# QuantumShieldProAPK

# Mobile Trading Terminal (APK) — Kivy
Esta carpeta contiene una versión **nativa móvil** del terminal (sin Streamlit), pensada para compilarse como **APK** usando **Kivy + python-for-android**.
## Qué incluye (MVP funcional)
- Pantalla tipo “terminal” con:
  - Ticker + timeframe
  - Recomendación (Compra/Venta/Neutral) con color
  - KPIs (Último, % cambio, RSI, ADX, ATR%)
  - Panel “Técnicos multi‑timeframe” (1D/4H/1H/15m) con consenso
- Datos de mercado desde **Yahoo Finance (endpoint JSON)**, sin `yfinance`.
- Indicadores calculados con **NumPy** (sin `pandas`), para que el APK sea viable.
## Requisitos de build (Windows)
Compilar APK desde Windows se hace normalmente con:
- **WSL2 (Ubuntu)** recomendado, o
- Linux nativo.
