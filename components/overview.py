"""
Overview tab component for the Streamlit application.
"""

import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt


def render_overview_tab(symbol: str):
    """Render the stock market overview tab."""

    st.header("Stock Market Overview")
    st.markdown(
        "Welcome to the Stock Prediction System! Select a stock symbol from the sidebar to view detailed market analysis.")

    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period="1mo")

            if not hist_data.empty:
                col1, col2, col3, col4 = st.columns(4)

                current_price = hist_data['Close'].iloc[-1]
                prev_price = hist_data['Close'].iloc[-2] if len(
                    hist_data) > 1 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100

                with col1:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                    )

                with col2:
                    st.metric("30-Day High",
                              f"${hist_data['High'].max():.2f}")

                with col3:
                    st.metric("30-Day Low",
                              f"${hist_data['Low'].min():.2f}")

                with col4:
                    avg_volume = hist_data['Volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")

                st.subheader(f"{symbol} - 30 Day Price Chart")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(hist_data.index,
                        hist_data['Close'], linewidth=2, color='#1f77b4')
                ax.fill_between(
                    hist_data.index, hist_data['Close'], alpha=0.3, color='#1f77b4')
                ax.set_title(f"{symbol} Stock Price - Last 30 Days")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.error(f"No data found for symbol: {symbol}")

        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
    else:
        st.info(
            "Please select a stock symbol from the sidebar to view market overview.")
