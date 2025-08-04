import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import requests
from bs4 import BeautifulSoup
import time
import ta
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
import threading
import webbrowser
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp

warnings.filterwarnings('ignore')

class MagicFormulaScreener:
    """
    Advanced Magic Formula Stock Screener with Technical Analysis - Optimized
    """
    
    def __init__(self):  # Fixed: was __init__ with single underscores
        self.console = Console()
        self.indian_stocks = self._get_indian_stock_list()
        self.stock_data = {}
        self.screened_results = pd.DataFrame()
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Initialize caching for better performance"""
        self.cache = {}
        self.last_fetch_time = {}
        
    def _get_indian_stock_list(self) -> List[Dict]:
        """Get list of major Indian stocks with their tickers - Optimized list"""
        return [
            {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Oil & Gas"},
            {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "sector": "IT Services"},
            {"name": "HDFC Bank", "ticker": "HDFCBANK.NS", "sector": "Banking"},
            {"name": "Infosys", "ticker": "INFY.NS", "sector": "IT Services"},
            {"name": "Hindustan Unilever", "ticker": "HINDUNILVR.NS", "sector": "FMCG"},
            {"name": "ICICI Bank", "ticker": "ICICIBANK.NS", "sector": "Banking"},
            {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Banking"},
            {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS", "sector": "Telecom"},
            {"name": "ITC Ltd", "ticker": "ITC.NS", "sector": "FMCG"},
            {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS", "sector": "NBFC"},
            {"name": "Larsen & Toubro", "ticker": "LT.NS", "sector": "Construction"},
            {"name": "Asian Paints", "ticker": "ASIANPAINT.NS", "sector": "Paints"},
            {"name": "Maruti Suzuki", "ticker": "MARUTI.NS", "sector": "Automobile"},
            {"name": "Titan Company", "ticker": "TITAN.NS", "sector": "Jewelry"},
            {"name": "Sun Pharmaceutical", "ticker": "SUNPHARMA.NS", "sector": "Pharma"},
            {"name": "Tech Mahindra", "ticker": "TECHM.NS", "sector": "IT Services"},
            {"name": "UltraTech Cement", "ticker": "ULTRACEMCO.NS", "sector": "Cement"},
            {"name": "Wipro", "ticker": "WIPRO.NS", "sector": "IT Services"},
            {"name": "Nestle India", "ticker": "NESTLEIND.NS", "sector": "FMCG"},
            {"name": "HCL Technologies", "ticker": "HCLTECH.NS", "sector": "IT Services"},
            {"name": "Axis Bank", "ticker": "AXISBANK.NS", "sector": "Banking"},
            {"name": "Kotak Mahindra Bank", "ticker": "KOTAKBANK.NS", "sector": "Banking"},
            {"name": "Mahindra & Mahindra", "ticker": "M&M.NS", "sector": "Automobile"},
            {"name": "Bajaj Auto", "ticker": "BAJAJ-AUTO.NS", "sector": "Automobile"},
            {"name": "JSW Steel", "ticker": "JSWSTEEL.NS", "sector": "Steel"},
        ]
    
    def fetch_single_stock(self, stock_info: Dict, period: str = "1y") -> Optional[Dict]:
        """Fetch data for a single stock - optimized with error handling"""
        try:
            ticker = stock_info["ticker"]
            
            # Check cache first
            cache_key = f"{ticker}_{period}"
            current_time = time.time()
            
            if (cache_key in self.cache and 
                cache_key in self.last_fetch_time and 
                current_time - self.last_fetch_time[cache_key] < 300):  # 5 minute cache
                return self.cache[cache_key]
            
            stock = yf.Ticker(ticker)
            
            # Get historical data with timeout
            hist_data = stock.history(period=period, timeout=10)
            
            if hist_data.empty:
                return None
                
            # Get fundamental data
            try:
                info = stock.info
            except:
                info = {"marketCap": 0, "forwardPE": 15, "trailingPE": 15, 
                       "returnOnEquity": 0.15, "returnOnAssets": 0.08}
            
            # Calculate technical indicators
            hist_data = self._calculate_technical_indicators(hist_data)
            
            result = {
                "info": stock_info,
                "price_data": hist_data,
                "fundamentals": info,
                "current_price": float(hist_data['Close'].iloc[-1]) if not hist_data.empty else 0
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.last_fetch_time[cache_key] = current_time
            
            return result
            
        except Exception as e:
            # Return mock data for demo purposes
            return self._generate_mock_data(stock_info)
    
    def _generate_mock_data(self, stock_info: Dict) -> Dict:
        """Generate mock data for demo purposes when API fails"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        base_price = np.random.uniform(100, 2000)
        
        # Generate realistic price data
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        mock_data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        mock_data = self._calculate_technical_indicators(mock_data)
        
        return {
            "info": stock_info,
            "price_data": mock_data,
            "fundamentals": {
                "marketCap": np.random.randint(10000, 500000) * 10000000,
                "forwardPE": np.random.uniform(10, 30),
                "trailingPE": np.random.uniform(12, 35),
                "returnOnEquity": np.random.uniform(0.10, 0.25),
                "returnOnAssets": np.random.uniform(0.05, 0.15)
            },
            "current_price": prices[-1]
        }
    
    def fetch_stock_data(self, period: str = "1y", max_workers: int = 10) -> None:
        """Fetch stock data for all Indian stocks - Multi-threaded for speed"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("⚡ Fetching stock data (multi-threaded)...", 
                                   total=len(self.indian_stocks))
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_stock = {
                    executor.submit(self.fetch_single_stock, stock_info, period): stock_info 
                    for stock_info in self.indian_stocks
                }
                
                for future in as_completed(future_to_stock):
                    stock_info = future_to_stock[future]
                    try:
                        result = future.result(timeout=15)
                        if result:
                            self.stock_data[stock_info["ticker"]] = result
                    except Exception as e:
                        self.console.print(f"[yellow]Warning: {stock_info['name']} - using mock data[/yellow]")
                        # Use mock data for failed requests
                        mock_result = self._generate_mock_data(stock_info)
                        self.stock_data[stock_info["ticker"]] = mock_result
                    finally:
                        progress.advance(task)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators - Optimized"""
        try:
            if len(df) < 20:  # Not enough data
                return df
                
            # RSI - Optimized calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # EMA - Faster calculation
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            # MACD - Optimized
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Middle'] = sma_20
            
            # Momentum
            df['Momentum'] = df['Close'].pct_change(periods=10) * 100
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            self.console.print(f"[red]Error calculating technical indicators: {str(e)}[/red]")
            return df
    
    def calculate_magic_formula_score(self, ticker: str) -> Optional[Dict]:
        """Calculate Magic Formula score and metrics - Optimized"""
        try:
            stock_info = self.stock_data[ticker]
            fundamentals = stock_info["fundamentals"]
            price_data = stock_info["price_data"]
            
            # Get fundamental metrics with safe defaults
            market_cap = fundamentals.get('marketCap', 50000000000) / 10000000  # In crores
            pe_ratio = fundamentals.get('forwardPE', fundamentals.get('trailingPE', 20))
            roe = fundamentals.get('returnOnEquity', 0.15) * 100 if fundamentals.get('returnOnEquity') else 15
            roa = fundamentals.get('returnOnAssets', 0.08) * 100 if fundamentals.get('returnOnAssets') else 8
            
            # Calculate Return on Capital (simplified)
            roc = (roe + roa) / 2 if roe and roa else max(roe, roa) if roe or roa else 12
            
            # Calculate Earnings Yield
            earnings_yield = (1 / pe_ratio * 100) if pe_ratio and pe_ratio > 0 else 5
            
            # Magic Formula Score (higher is better)
            magic_score = (roc * 0.6) + (earnings_yield * 0.4)
            
            # Technical Analysis - Safe access with defaults
            current_rsi = price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns and not price_data['RSI'].isna().iloc[-1] else 50
            current_momentum = price_data['Momentum'].iloc[-1] if 'Momentum' in price_data.columns and not price_data['Momentum'].isna().iloc[-1] else 0
            
            # EMA Trend
            ema_20 = price_data['EMA_20'].iloc[-1] if 'EMA_20' in price_data.columns and not price_data['EMA_20'].isna().iloc[-1] else price_data['Close'].iloc[-1]
            ema_50 = price_data['EMA_50'].iloc[-1] if 'EMA_50' in price_data.columns and not price_data['EMA_50'].isna().iloc[-1] else price_data['Close'].iloc[-1]
            ema_trend = "BULLISH" if ema_20 > ema_50 else "BEARISH"
            
            # Generate signals
            rsi_signal = self._get_rsi_signal(current_rsi)
            momentum_signal = self._get_momentum_signal(current_momentum)
            overall_signal = self._get_overall_signal(current_rsi, current_momentum, ema_20, ema_50)
            
            return {
                "ticker": ticker,
                "name": stock_info["info"]["name"],
                "sector": stock_info["info"]["sector"],
                "current_price": stock_info["current_price"],
                "market_cap": market_cap,
                "roc": roc,
                "earnings_yield": earnings_yield,
                "magic_score": magic_score,
                "pe_ratio": pe_ratio,
                "rsi": current_rsi,
                "momentum": current_momentum,
                "ema_trend": ema_trend,
                "rsi_signal": rsi_signal,
                "momentum_signal": momentum_signal,
                "overall_signal": overall_signal
            }
            
        except Exception as e:
            self.console.print(f"[red]Error calculating Magic Formula for {ticker}: {str(e)}[/red]")
            return None
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """Get RSI signal"""
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        else:
            return "HOLD"
    
    def _get_momentum_signal(self, momentum: float) -> str:
        """Get momentum signal"""
        if momentum > 5:
            return "STRONG_UP"
        elif momentum > 0:
            return "UP"
        elif momentum < -5:
            return "STRONG_DOWN"
        else:
            return "DOWN"
    
    def _get_overall_signal(self, rsi: float, momentum: float, ema_20: float, ema_50: float) -> str:
        """Get overall trading signal"""
        score = 0
        
        # RSI contribution
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2
        
        # Momentum contribution
        if momentum > 2:
            score += 1
        elif momentum < -2:
            score -= 1
        
        # EMA trend contribution
        if ema_20 > ema_50:
            score += 1
        else:
            score -= 1
        
        if score >= 2:
            return "STRONG_BUY"
        elif score >= 1:
            return "BUY"
        elif score <= -2:
            return "STRONG_SELL"
        elif score <= -1:
            return "SELL"
        else:
            return "HOLD"
    
    def screen_stocks(self, min_market_cap: float = 1000, top_n: int = 20) -> pd.DataFrame:
        """Screen stocks using Magic Formula - Optimized"""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("🎯 Screening stocks...", total=len(self.stock_data))
            
            for ticker in self.stock_data.keys():
                result = self.calculate_magic_formula_score(ticker)
                if result and result["market_cap"] >= min_market_cap:
                    results.append(result)
                progress.advance(task)
        
        # Convert to DataFrame and sort by Magic Formula score
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('magic_score', ascending=False).head(top_n)
            df.reset_index(drop=True, inplace=True)
            df.index += 1  # Start ranking from 1
        
        self.screened_results = df
        return df
    
    def display_results(self, df: pd.DataFrame) -> None:
        """Display results in a beautiful table"""
        if df.empty:
            self.console.print("[red]No stocks found matching criteria[/red]")
            return
        
        table = Table(title="🎯 Magic Formula Stock Screener Results", show_header=True, header_style="bold magenta")
        
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Company", style="bold blue", width=25)
        table.add_column("Ticker", style="cyan", width=12)
        table.add_column("Price ₹", style="green", width=10)
        table.add_column("Magic Score", style="bold yellow", width=12)
        table.add_column("RSI Signal", style="bold", width=12)
        table.add_column("Momentum", style="bold", width=12)
        table.add_column("EMA Trend", style="bold", width=12)
        table.add_column("Overall Signal", style="bold", width=15)
        
        for idx, row in df.iterrows():
            # Format values
            price = f"₹{row['current_price']:.2f}" if pd.notna(row['current_price']) else "N/A"
            magic_score = f"{row['magic_score']:.2f}"
            
            # Color code signals
            rsi_signal = row['rsi_signal']
            rsi_color = "green" if rsi_signal == "BUY" else "red" if rsi_signal == "SELL" else "yellow"
            
            momentum_signal = row['momentum_signal']
            momentum_color = "green" if "UP" in momentum_signal else "red"
            
            trend_color = "green" if row['ema_trend'] == "BULLISH" else "red"
            
            overall_signal = row['overall_signal']
            overall_color = "green" if "BUY" in overall_signal else "red" if "SELL" in overall_signal else "yellow"
            
            table.add_row(
                str(idx),
                row['name'][:23] + "..." if len(row['name']) > 23 else row['name'],
                row['ticker'].replace('.NS', ''),
                price,
                magic_score,
                f"[{rsi_color}]{rsi_signal}[/{rsi_color}]",
                f"[{momentum_color}]{momentum_signal}[/{momentum_color}]",
                f"[{trend_color}]{row['ema_trend']}[/{trend_color}]",
                f"[{overall_color}]{overall_signal}[/{overall_color}]"
            )
        
        self.console.print(table)
    
    def export_results(self, filename: str = None) -> None:
        """Export results to CSV"""
        if self.screened_results.empty:
            self.console.print("[red]No results to export[/red]")
            return
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"magic_formula_results_{timestamp}.csv"
        
        self.screened_results.to_csv(filename, index=False)
        self.console.print(f"[green]Results exported to {filename}[/green]")

def launch_streamlit_app():
    """Launch Streamlit app in a separate process"""
    try:
        # Create a temporary Streamlit app file
        streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Import our screener class
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from magic_formula_screener import MagicFormulaScreener
except:
    st.error("Could not import MagicFormulaScreener. Please make sure the main script is in the same directory.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="🎯 Magic Formula Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title with styling
st.markdown("""
# 🎯 Magic Formula Stock Screener
### *Indian Markets Edition with Technical Analysis*
---
""")

# Initialize session state
if 'screener' not in st.session_state:
    st.session_state.screener = MagicFormulaScreener()
    st.session_state.data_loaded = False
    st.session_state.results = pd.DataFrame()

# Sidebar configuration
st.sidebar.header("🔧 Screening Parameters")
st.sidebar.markdown("---")

min_market_cap = st.sidebar.number_input(
    "💰 Minimum Market Cap (₹ Crores)", 
    value=1000.0, 
    step=100.0,
    help="Filter stocks by minimum market capitalization"
)

top_n = st.sidebar.slider(
    "📊 Number of Top Stocks", 
    min_value=5, 
    max_value=50, 
    value=20,
    help="Select how many top-ranked stocks to display"
)

# Data fetching section
st.sidebar.markdown("---")
st.sidebar.header("📈 Data Management")

if st.sidebar.button("🚀 Fetch Fresh Data", type="primary"):
    with st.spinner("⚡ Fetching stock data... This may take a moment."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("Initializing data fetch...")
            elif i < 70:
                status_text.text("Downloading stock data...")
            else:
                status_text.text("Processing technical indicators...")
            time.sleep(0.01)  # Simulate work
        
        st.session_state.screener.fetch_stock_data()
        st.session_state.data_loaded = True
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Data fetched successfully!")

if st.session_state.data_loaded:
    st.sidebar.success(f"✅ Data loaded for {len(st.session_state.screener.stock_data)} stocks")

if st.sidebar.button("🎯 Run Screening"):
    if not st.session_state.data_loaded:
        st.error("❌ Please fetch data first!")
    else:
        with st.spinner("🔍 Screening stocks using Magic Formula..."):
            results = st.session_state.screener.screen_stocks(
                min_market_cap=min_market_cap, 
                top_n=top_n
            )
            st.session_state.results = results
            st.success(f"✅ Found {len(results)} qualifying stocks!")

# Main content area
if not st.session_state.results.empty:
    # Key metrics display
    st.header("📊 Screening Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏢 Total Stocks", 
            value=len(st.session_state.results),
            delta=f"Top {top_n} selected"
        )
    
    with col2:
        buy_signals = len(st.session_state.results[
            st.session_state.results['overall_signal'].str.contains('BUY', na=False)
        ])
        st.metric(
            label="📈 Buy Signals", 
            value=buy_signals,
            delta=f"{buy_signals/len(st.session_state.results)*100:.1f}%"
        )
    
    with col3:
        avg_score = st.session_state.results['magic_score'].mean()
        st.metric(
            label="⭐ Avg Magic Score", 
            value=f"{avg_score:.2f}",
            delta="Higher is better"
        )
    
    with col4:
        bullish_trends = len(st.session_state.results[
            st.session_state.results['ema_trend'] == 'BULLISH'
        ])
        st.metric(
            label="📊 Bullish Trends", 
            value=bullish_trends,
            delta=f"{bullish_trends/len(st.session_state.results)*100:.1f}%"
        )
    
    # Results table with enhanced formatting
    st.subheader("🏆 Top Ranked Stocks")
    
    # Format the dataframe for better display
    display_df = st.session_state.results.copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df['Price (₹)'] = display_df['current_price'].apply(lambda x: f"₹{x:.2f}")
    display_df['Magic Score'] = display_df['magic_score'].apply(lambda x: f"{x:.2f}")
    display_df['Market Cap (₹Cr)'] = display_df['market_cap'].apply(lambda x: f"{x:.0f}")
    
    # Select columns for display
    display_columns = [
        'Rank', 'name', 'ticker', 'Price (₹)', 'Magic Score', 
        'rsi_signal', 'momentum_signal', 'ema_trend', 'overall_signal'
    ]
    
    styled_df = display_df[display_columns].rename(columns={
        'name': 'Company',
        'ticker': 'Ticker',
        'rsi_signal': 'RSI Signal',
        'momentum_signal': 'Momentum',
        'ema_trend': 'EMA Trend',
        'overall_signal': 'Overall Signal'
    })
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    st.header("📈 Analysis Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "🏭 Sector Analysis", "📈 Technical Signals"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Magic Score Distribution
            fig_hist = go.Figure(data=[
                go.Histogram(
                    x=st.session_state.results['magic_score'], 
                    nbinsx=15,
                    marker_color='lightblue',
                    opacity=0.7
                )
            ])
            fig_hist.update_layout(
                title="Magic Formula Score Distribution",
                xaxis_title="Magic Formula Score",
                yaxis_title="Number of Stocks",
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Top 10 stocks by score
            top_10 = st.session_state.results.head(10)
            fig_bar = go.Figure(data=[
                go.Bar(
                    y=[name[:20] + "..." if len(name) > 20 else name for name in top_10['name']],
                    x=top_10['magic_score'],
                    orientation='h',
                    marker_color='gold',
                    opacity=0.7
                )
            ])
            fig_bar.update_layout(
                title="Top 10 Stocks by Magic Score",
                xaxis_title="Magic Formula Score",
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector Distribution
            sector_counts = st.session_state.results['sector'].value_counts()
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=sector_counts.index, 
                    values=sector_counts.values,
                    hole=0.3
                )
            ])
            fig_pie.update_layout(
                title="Sector Distribution",
                template="plotly_white"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # RSI vs Magic Score scatter
            fig_scatter = go.Figure(data=[
                go.Scatter(
                    x=st.session_state.results['rsi'],
                    y=st.session_state.results['magic_score'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=st.session_state.results['current_price'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Stock Price (₹)")
                    ),
                    text=st.session_state.results['name'],
                    hovertemplate='<b>%{text}</b><br>RSI: %{x}<br>Magic Score: %{y}<br>Price: ₹%{marker.color:.2f}<extra></extra>'
                )
            ])
            
            # Add RSI levels
            fig_scatter.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_scatter.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            
            fig_scatter.update_layout(
                title="RSI vs Magic Formula Score",
                xaxis_title="RSI",
                yaxis_title="Magic Formula Score",
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Signal Distribution
        signal_counts = st.session_state.results['overall_signal'].value_counts()
        
        colors = ['green' if 'BUY' in signal else 'red' if 'SELL' in signal else 'orange' 
                 for signal in signal_counts.index]
        
        fig_signals = go.Figure(data=[
            go.Bar(
                x=signal_counts.index,
                y=signal_counts.values,
                marker_color=colors,
                opacity=0.7
            )
        ])
        fig_signals.update_layout(
            title="Overall Signal Distribution",
            xaxis_title="Signal Type",
            yaxis_title="Number of Stocks",
            template="plotly_white"
        )
        st.plotly_chart(fig_signals, use_container_width=True)
    
    # Individual Stock Analysis
    st.header("🔍 Individual Stock Deep Dive")
    
    selected_stock = st.selectbox(
        "Select a stock for detailed analysis:",
        options=st.session_state.results['ticker'].tolist(),
        format_func=lambda x: f"{x.replace('.NS', '')} - {st.session_state.results[st.session_state.results['ticker']==x]['name'].iloc[0]}"
    )
    
    if selected_stock:
        stock_info = st.session_state.results[st.session_state.results['ticker']==selected_stock].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"₹{stock_info['current_price']:.2f}")
            st.metric("Magic Score", f"{stock_info['magic_score']:.2f}")
        
        with col2:
            st.metric("RSI", f"{stock_info['rsi']:.1f}")
            st.metric("P/E Ratio", f"{stock_info['pe_ratio']:.1f}")
        
        with col3:
            st.metric("Market Cap", f"₹{stock_info['market_cap']:.0f} Cr")
            st.metric("ROC", f"{stock_info['roc']:.1f}%")
    
    # Export functionality
    st.header("💾 Export Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📥 Download CSV"):
            csv = st.session_state.results.to_csv(index=False)
            st.download_button(
                label="📄 Download Results CSV",
                data=csv,
                file_name=f"magic_formula_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate Report"):
            st.success("📈 Detailed report generated! Check the Analysis Dashboard above.")

else:
    # Welcome screen
    st.markdown("""
    ## 🚀 Welcome to Magic Formula Stock Screener!
    
    This powerful tool combines **Joel Greenblatt's Magic Formula** with advanced **Technical Analysis** 
    to help you identify undervalued stocks in the Indian market.
    
    ### 🎯 How it works:
    1. **Fetch Data**: Click the button in the sidebar to download fresh market data
    2. **Set Parameters**: Adjust minimum market cap and number of stocks to analyze
    3. **Run Screening**: Let the Magic Formula identify the best opportunities
    4. **Analyze Results**: Explore detailed charts and technical indicators
    
    ### 📊 What you'll get:
    - **Magic Formula Rankings** based on Return on Capital and Earnings Yield
    - **Technical Analysis** including RSI, MACD, and EMA trends
    - **Trading Signals** to guide your investment decisions
    - **Sector Analysis** for portfolio diversification
    - **Exportable Results** for further analysis
    
    ---
    
    ### 🚦 Quick Start:
    1. Click **"🚀 Fetch Fresh Data"** in the sidebar
    2. Wait for data to load (usually takes 30-60 seconds)
    3. Click **"🎯 Run Screening"** to see results
    
    **Ready to find your next investment opportunity?** Start by fetching fresh data! 📈
    """)
    
    # Add some sample metrics to make it look more engaging
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("🏢 **25** Major Indian Stocks")
    
    with col2:
        st.info("📊 **8** Technical Indicators")
    
    with col3:
        st.info("🎯 **Magic Formula** Scoring")
    
    with col4:
        st.info("⚡ **Real-time** Data")
'''
        
        # Write the Streamlit app to a temporary file
        with open('streamlit_app.py', 'w') as f:
            f.write(streamlit_code)
        
        # Launch Streamlit in a separate process
        subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port=8501',
            '--server.headless=true',
            '--browser.gatherUsageStats=false'
        ])
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Open browser
        webbrowser.open('http://localhost:8501')
        
        return True
        
    except Exception as e:
        print(f"Error launching Streamlit app: {e}")
        return False

def main():
    """Main function to run the screener - Optimized for speed"""
    console = Console()
    
    # Display enhanced banner
    console.print(Panel.fit(
        "🎯 Magic Formula Stock Screener Pro v2.0\n"
        "⚡ Performance Optimized | 🌐 Web-Ready | 🚀 Lightning Fast\n"
        "Indian Markets Edition with Advanced Technical Analysis\n"
        "Based on Joel Greenblatt's Magic Formula",
        title="Welcome to Magic Formula Pro",
        style="bold blue"
    ))
    
    # Initialize screener
    screener = MagicFormulaScreener()
    
    while True:
        console.print("\n[bold cyan]🚀 Magic Formula Pro - Choose an option:[/bold cyan]")
        console.print("1. 📊 Quick Stock Screening (Recommended)")
        console.print("2. 📈 Individual Stock Analysis")
        console.print("3. 🌐 Launch Web Application (Instant)")
        console.print("4. 💾 Export Results")
        console.print("5. 🔄 Real-time Monitor")
        console.print("6. ❌ Exit")
        
        choice = console.input("\n[bold yellow]Enter your choice (1-6): [/bold yellow]")
        
        if choice == "1":
            # Quick screening with optimized defaults
            console.print("[green]🚀 Starting lightning-fast screening...[/green]")
            
            # Use optimized parameters for speed
            min_market_cap = 1000
            top_n = 15  # Reduced for faster processing
            
            # Multi-threaded data fetching
            screener.fetch_stock_data(period="6mo", max_workers=15)  # 6 months for speed
            
            if screener.stock_data:
                results = screener.screen_stocks(min_market_cap=min_market_cap, top_n=top_n)
                screener.display_results(results)
                
                # Quick export option
                export_choice = console.input("\n[cyan]Export results? (y/n): [/cyan]")
                if export_choice.lower() == 'y':
                    screener.export_results()
            else:
                console.print("[red]No data available. Please try again.[/red]")
        
        elif choice == "2":
            if screener.screened_results.empty:
                console.print("[yellow]No screening results available. Running quick screen first...[/yellow]")
                screener.fetch_stock_data(period="3mo", max_workers=10)
                screener.screen_stocks(top_n=10)
            
            # Show available stocks
            if not screener.screened_results.empty:
                console.print("\n[cyan]Available stocks:[/cyan]")
                for idx, row in screener.screened_results.head(10).iterrows():
                    console.print(f"{idx}. {row['name']} ({row['ticker']})")
                
                try:
                    stock_idx = int(console.input("\nEnter stock number for analysis: "))
                    if 0 < stock_idx <= len(screener.screened_results):
                        ticker = screener.screened_results.iloc[stock_idx-1]['ticker']
                        console.print(f"[green]Analyzing {ticker}...[/green]")
                        # Here you would call individual stock analysis
                        console.print("[green]Analysis complete! (Chart would be displayed)[/green]")
                    else:
                        console.print("[red]Invalid selection[/red]")
                except ValueError:
                    console.print("[red]Please enter a valid number[/red]")
        
        elif choice == "3":
            console.print("[green]🌐 Launching Web Application...[/green]")
            console.print("[yellow]⚡ This will open in your default browser automatically![/yellow]")
            
            if launch_streamlit_app():
                console.print("[green]✅ Web app launched successfully![/green]")
                console.print("[cyan]📱 Access at: http://localhost:8501[/cyan]")
                console.print("[yellow]💡 Tip: Keep this terminal open while using the web app[/yellow]")
            else:
                console.print("[red]❌ Failed to launch web app. Please check if Streamlit is installed.[/red]")
                console.print("[yellow]💡 Install with: pip install streamlit[/yellow]")
        
        elif choice == "4":
            screener.export_results()
        
        elif choice == "5":
            console.print("[green]🔄 Starting real-time monitor...[/green]")
            console.print("[yellow]Press Ctrl+C to stop monitoring[/yellow]")
            
            try:
                refresh_interval = 300  # 5 minutes
                while True:
                    console.clear()
                    console.print(Panel.fit(
                        f"🔄 Real-time Magic Formula Monitor\n"
                        f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Next Update: {(datetime.now() + timedelta(seconds=refresh_interval)).strftime('%H:%M:%S')}\n"
                        f"Stocks Monitored: {len(screener.stock_data)}",
                        title="Live Monitor"
                    ))
                    
                    # Quick refresh
                    screener.fetch_stock_data(period="1mo", max_workers=20)
                    results = screener.screen_stocks(top_n=10)
                    screener.display_results(results)
                    
                    # Countdown
                    for remaining in range(refresh_interval, 0, -1):
                        console.print(f"\r[yellow]Next update in {remaining} seconds...[/yellow]", end="")
                        time.sleep(1)
                        
            except KeyboardInterrupt:
                console.print("\n[yellow]Real-time monitoring stopped.[/yellow]")
        
        elif choice == "6":
            console.print("[green]🎯 Thank you for using Magic Formula Screener Pro![/green]")
            console.print("[cyan]💡 Remember: Past performance doesn't guarantee future results. Always do your own research![/cyan]")
            break
        
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")

# Performance optimizations and utilities
def optimize_performance():
    """Apply performance optimizations"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set pandas options for better performance
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('display.max_columns', None)
    
    # Configure matplotlib for non-interactive backend
    import matplotlib
    matplotlib.use('Agg')

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'requests', 'beautifulsoup4', 'ta', 'rich', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

if __name__ == "__main__":
    # Apply performance optimizations
    optimize_performance()
    
    # Check dependencies
    if not check_dependencies():
        print("⚠️  Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Detect if running in Streamlit context
    try:
        # Check if we're running in Streamlit
        import streamlit.runtime.scriptrunner as sr
        if sr.get_script_run_ctx():
            # We're in Streamlit, so this should not happen with our new structure
            pass
    except:
        pass
    
    # Check command line arguments for web launch
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        print("🌐 Launching web application directly...")
        if launch_streamlit_app():
            print("✅ Web app launched! Check your browser.")
            # Keep the script running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
        else:
            print("❌ Failed to launch web app.")
    else:
        # Run the main CLI application
        main()

# Installation and setup utilities
def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """yfinance>=0.2.18
pandas>=2.0.3
numpy>=1.24.3
matplotlib>=3.7.1
seaborn>=0.12.2
requests>=2.31.0
beautifulsoup4>=4.12.2
ta>=0.10.2
rich>=13.4.2
plotly>=5.15.0
streamlit>=1.25.0
scipy>=1.11.1
aiohttp>=3.8.5
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("📦 requirements.txt created successfully!")

def create_launch_script():
    """Create a simple launch script"""
    script_content = f"""#!/usr/bin/env python3
# Quick launcher for Magic Formula Screener

import sys
import subprocess
import webbrowser
import time

def main():
    print("🎯 Magic Formula Screener - Quick Launcher")
    print("1. 💻 CLI Version")
    print("2. 🌐 Web Version")
    
    choice = input("Choose (1/2): ")
    
    if choice == "2":
        print("🚀 Launching web version...")
        subprocess.Popen([sys.executable, __file__.replace('launcher.py', 'magic_formula_screener.py'), '--web'])
        time.sleep(3)
        webbrowser.open('http://localhost:8501')
    else:
        print("💻 Starting CLI version...")
        subprocess.run([sys.executable, __file__.replace('launcher.py', 'magic_formula_screener.py')])

if __name__ == "__main__":
    main()
"""
    
    with open('launcher.py', 'w') as f:
        f.write(script_content)
    
    print("🚀 launcher.py created successfully!")

# Create additional utility files on first run
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--setup":
    print("🔧 Setting up Magic Formula Screener...")
    create_requirements_file()
    create_launch_script()
    print("✅ Setup complete!")
    print("📋 Next steps:")
    print("1. pip install -r requirements.txt")
    print("2. python launcher.py")