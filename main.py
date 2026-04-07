from __future__ import annotations

import threading

from kivy.app import App
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from core.engine import compute_all, consensus, recommend, snapshot, technical_signals
from core.yahoo_chart import YahooError, fetch_ohlcv


TF_MAP = {
    "1D": ("1y", "1d"),
    "4H": ("1y", "4h"),
    "1H": ("60d", "1h"),
    "15m": ("30d", "15m"),
}


def _hex_to_rgba01(h: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = (h or "").lstrip("#")
    if len(h) != 6:
        return (0.55, 0.58, 0.62, a)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b, a)


class RootUI(BoxLayout):
    def on_refresh(self) -> None:
        app = App.get_running_app()
        if app:
            app.refresh_async()


class QuantumShieldMobile(App):
    root_widget: RootUI = ObjectProperty(None)

    def build(self):
        self.title = "QuantumShield Mobile"
        self.root_widget = Builder.load_file("ui.kv")
        # first load
        self.refresh_async()
        return self.root_widget

    def refresh_async(self) -> None:
        ticker = self.root_widget.ids.ticker.text.strip().upper()
        tf = self.root_widget.ids.tf.text
        self.root_widget.ids.status.text = "Cargando…"

        def worker():
            try:
                rng, interval = TF_MAP.get(tf, ("1y", "1d"))
                ohlcv = fetch_ohlcv(ticker, range_=rng, interval=interval)

                feats = compute_all(ohlcv)
                snap = snapshot(ohlcv, feats)
                rec = recommend(ohlcv, feats)

                # Build multi-timeframe consensus (fixed list)
                tf_rows = []
                for tfl in ["1D", "4H", "1H", "15m"]:
                    rng2, interval2 = TF_MAP[tfl]
                    try:
                        o2 = fetch_ohlcv(ticker, range_=rng2, interval=interval2)
                        f2 = compute_all(o2)
                        close2 = float(o2.c[-1])
                        ma, osc = technical_signals(close2, f2)
                        cons = consensus(ma, osc)
                        tf_rows.append((tfl, cons["Consenso"], cons["Net"], cons["MA Compra"], cons["MA Venta"], cons["Osc Compra"], cons["Osc Venta"]))
                    except Exception:
                        tf_rows.append((tfl, "—", "—", "—", "—", "—", "—"))

                self._update_ui(ticker, tf, snap, rec, tf_rows)
            except YahooError as e:
                self._set_status(f"Error datos: {e}")
            except Exception as e:
                self._set_status(f"Error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _set_status(self, text: str) -> None:
        from kivy.clock import Clock

        def apply(_dt):
            self.root_widget.ids.status.text = text

        Clock.schedule_once(apply, 0)

    def _update_ui(self, ticker: str, tf: str, snap, rec, tf_rows) -> None:
        from kivy.clock import Clock

        def apply(_dt):
            # Recommendation card coloring
            border = _hex_to_rgba01(rec.color_hex, 1.0)
            bg = _hex_to_rgba01(rec.color_hex, 0.16)
            self.root_widget.ids.rec_card.border_rgba = border
            self.root_widget.ids.rec_card.bg_rgba = bg

            self.root_widget.ids.rec_value.text = rec.label
            self.root_widget.ids.rec_value.color = _hex_to_rgba01(rec.color_hex, 1.0)
            self.root_widget.ids.rec_sub.text = f"Score {rec.score:+.0f}/100 • Confianza {rec.confidence}% • Régimen {rec.regime}"

            chg = "—" if snap.change_pct is None else f"{snap.change_pct:+.2f}%"
            self.root_widget.ids.kpi_price.text = f"Último: {snap.close:.4f}".rstrip("0").rstrip(".") + f"   ({chg})"
            self.root_widget.ids.kpi_rsi.text = "RSI14: —" if snap.rsi14 is None else f"RSI14: {snap.rsi14:.1f}"
            self.root_widget.ids.kpi_adx.text = "ADX14: —" if snap.adx14 is None else f"ADX14: {snap.adx14:.1f} ({rec.regime})"
            self.root_widget.ids.kpi_atr.text = "ATR%: —" if snap.atrp14 is None else f"ATR%: {snap.atrp14:.2f}%"

            grid = self.root_widget.ids.tf_grid
            grid.clear_widgets()

            header = Label(
                text="[b]TF[/b]   [b]Consenso[/b]   Net   MA(B/S)   Osc(B/S)",
                markup=True,
                size_hint_y=None,
                height=dp(22),
                halign="left",
                valign="middle",
                color=(0.9, 0.92, 0.95, 0.9),
            )
            header.bind(size=lambda *_: setattr(header, "text_size", header.size))
            grid.add_widget(header)

            for tfl, cons, net, bma, sma, bosc, sosc in tf_rows:
                row = Label(
                    text=f"{tfl:>3}   {cons:<12}   {net:>3}   {bma}/{sma}     {bosc}/{sosc}",
                    size_hint_y=None,
                    height=dp(22),
                    halign="left",
                    valign="middle",
                    color=(0.9, 0.92, 0.95, 0.85),
                )
                row.bind(size=lambda *_: setattr(row, "text_size", row.size))
                grid.add_widget(row)

            self.root_widget.ids.status.text = f"{ticker} • {tf} • OK"

        Clock.schedule_once(apply, 0)


if __name__ == "__main__":
    QuantumShieldMobile().run()

