"""
Top Stocks tab component for the Streamlit application.
"""

import streamlit as st
from .tools import get_popular_stocks, get_category_stocks_analysis, format_currency


def render_top_stocks_tab():
    """Render the top stocks tab with category selection."""

    st.header("🔥 Top 10 Stocks")
    st.markdown(
        "Discover the most popular and performing stocks based on different categories and metrics.")

    popular_stocks = get_popular_stocks()
    categories = list(popular_stocks.keys())

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_category = st.selectbox(
            "Select Stock Category:",
            index=0,
            options=categories,
            help="Choose a category to see the top 10 stocks"
        )

    with col2:
        analysis_type = st.selectbox(
            "Analysis Type:",
            index=0,
            options=["Popularity Score",
                     "Price Change", "Volume", "Volatility"],
            help="Choose how to rank the stocks"
        )

    if selected_category:
        st.info(f"📊 Showing top stocks from: **{selected_category}**")

        category_stocks = popular_stocks[selected_category]

        if category_stocks:
            with st.spinner(f"Analyzing {len(category_stocks)} stocks from {selected_category}..."):
                analysis_df = get_category_stocks_analysis(
                    category_stocks, analysis_type)

                if analysis_df is not None and not analysis_df.empty:
                    display_summary_metrics(analysis_df)

                    display_top_stocks_table(analysis_df, analysis_type)

                    display_top_performers(analysis_df)

                else:
                    st.warning(
                        "Could not fetch analysis data for the selected category.")
        else:
            st.warning("No stocks found in the selected category.")


def display_summary_metrics(df):
    """Display summary metrics for the category."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyzed", len(df))

    with col2:
        avg_change = df['price_change_pct'].mean()
        st.metric("Avg Monthly Change", f"{avg_change:.2f}%")

    with col3:
        total_volume = df['avg_volume'].sum()
        st.metric("Total Volume", f"{total_volume/1e9:.1f}B")

    with col4:
        avg_volatility = df['volatility'].mean()
        st.metric("Avg Volatility", f"{avg_volatility:.2f}%")


def display_top_stocks_table(df, analysis_type):
    """Display the top stocks in a styled table."""
    st.subheader(f"📈 Top 10 Stocks (Ranked by {analysis_type})")

    display_df = df.copy()
    display_df['price_change_pct'] = display_df['price_change_pct'].round(2)
    display_df['current_price'] = display_df['current_price'].round(2)
    display_df['volatility'] = display_df['volatility'].round(2)
    display_df['market_cap'] = display_df['market_cap'].apply(format_currency)
    display_df['avg_volume'] = (display_df['avg_volume'] / 1e6).round(1)
    display_df['popularity_score'] = display_df['popularity_score'].round(2)

    display_df = display_df.rename(columns={
        'symbol': 'Symbol',
        'name': 'Company',
        'sector': 'Sector',
        'price_change_pct': 'Monthly Change (%)',
        'current_price': 'Current Price ($)',
        'volatility': 'Volatility (%)',
        'market_cap': 'Market Cap',
        'avg_volume': 'Avg Volume (M)',
        'popularity_score': 'Popularity Score'
    })

    def highlight_change(val):
        if isinstance(val, (int, float)) and 'Change' in str(val):
            if val > 0:
                return 'background-color: #064E3B; color: #10B981'
            elif val < 0:
                return 'background-color: #7F1D1D; color: #EF4444'
        return ''

    styled_df = display_df.style.applymap(
        highlight_change, subset=['Monthly Change (%)'])
    st.dataframe(styled_df, width="content", hide_index=True)


def display_top_performers(df):
    """Display top 3 performers."""

    st.subheader("🏆 Top 3 Performers")

    top_3 = df.nlargest(3, 'price_change_pct')

    cols = st.columns(3)
    for i, (_, stock) in enumerate(top_3.iterrows()):
        with cols[i]:
            change_color = "🟢" if stock['price_change_pct'] > 0 else "🔴"
            st.metric(
                f"{change_color} {stock['symbol']}",
                f"${stock['current_price']:.2f}",
                f"{stock['price_change_pct']:.2f}%"
            )
            st.caption(f"{stock['name']}")
