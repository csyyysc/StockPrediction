# 📈 Multi-Agent Stock Predictor

A sophisticated stock prediction and trading application supporting multiple **reinforcement learning algorithms** including Vanilla Actor-Critic (VAC) and A3C (Asynchronous Advantage Actor-Critic). This system learns optimal trading strategies by analyzing historical stock data and making buy/hold/sell decisions to maximize portfolio returns.

## 🌟 Features

### 🚀 Core Capabilities
- **Multiple RL Algorithms**: Choose between VAC and A3C for different trading strategies
  - **VAC (Vanilla Actor-Critic)**: Classic synchronous algorithm with shared feature extraction
  - **A3C (Asynchronous Advantage Actor-Critic)**: Advanced algorithm with improved exploration and GAE
- **Customizable Training Windows**: 10-120 days of historical data as input features
- **Multi-Stock Support**: Trade any stock symbol with real-time data from Yahoo Finance
- **Risk Management**: Built-in portfolio management with cash and position tracking
- **Performance Analytics**: Comprehensive metrics including Sharpe ratio, max drawdown, and excess returns

### 🎯 Trading Actions
- **Hold** (0): Maintain current position
- **Buy** (1): Purchase stocks with available cash
- **Sell** (2): Liquidate all held positions

### 📊 Technical Indicators
- Price returns and volatility metrics
- Moving averages (5, 10, 20 days)
- Relative Strength Index (RSI)
- Bollinger Bands positioning
- Volume analysis
- Price momentum indicators

### 💻 Interface Options
- **Web Interface**: Beautiful Streamlit dashboard with real-time training visualization
- **Command Line**: Direct training and evaluation for automation
- **Model Persistence**: Save and load trained models for later use
- **Modular Architecture**: Clean, maintainable component-based design for easy extension

## 🏗️ Architecture

```
📦 Multi-Agent Application
├── 🧠 Agent Module
│   ├── BaseAgent (Abstract Interface)
│   ├── VAC Agent (Vanilla Actor-Critic)
│   └── A3C Agent (Asynchronous Advantage Actor-Critic)
│
├── 🌍 Stock Trading Environment
│   ├── Data Fetching (Yahoo Finance)
│   ├── Feature Engineering
│   ├── Portfolio Management
│   └── Reward Calculation
│
├── 🎓 Training System
│   ├── Multi-Agent Support
│   ├── Episode Management
│   ├── Performance Tracking
│   └── Model Checkpointing
│
├── 🖥️ User Interfaces
│   ├── Streamlit Web App (Modular Components)
│   └── Command Line Interface (with Agent Options)
│
└── 🧩 Component Architecture
    ├── Sidebar Configuration
    ├── Overview Dashboard
    ├── Training Interface
    ├── Results Visualization
    └── Analysis Tools
```

## 🧩 Modular Component Architecture

The application features a **clean, modular component architecture** that separates concerns and makes the codebase highly maintainable:

### 🎯 Component Benefits:
- **📦 Modularity**: Each UI section is a separate, focused component
- **🔧 Maintainability**: Easy to modify specific features without affecting others
- **🧪 Testability**: Components can be tested independently
- **♻️ Reusability**: Components can be imported and used in other projects
- **📈 Scalability**: Simple to add new tabs or modify existing functionality

### 🏗️ Component Structure:
- **`components/sidebar.py`**: Configuration controls and parameter selection
- **`components/overview.py`**: Stock market data display and charts
- **`components/training.py`**: Training interface and progress tracking
- **`components/results.py`**: Results visualization and performance metrics
- **`components/analysis.py`**: Detailed analysis and trading insights
- **`components/tools.py`**: Shared utility functions and data processing

### 🚀 Easy Extension:
```python
# Adding a new component is simple
from components import render_sidebar

def my_custom_tab():
    st.header("My Custom Analysis")
    # Your custom logic here

# Integrate into main app
with st.tabs(["📊 Overview", "🤖 Training", "📈 Results", "🔍 Analysis", "🆕 Custom"]):
    with st.tab("🆕 Custom"):
        my_custom_tab()
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd StockPrediction

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Launch Web Interface (Recommended)

```bash
# Using uv (recommended)
uv run python main.py

# Or directly with Python
python main.py
```

This opens an interactive Streamlit dashboard at `http://localhost:8501` with a **modular component architecture**:

#### 🧩 Web Interface Components:
- **📊 Overview Tab**: Real-time stock market data and price charts
- **🤖 Training Tab**: Interactive model training with progress tracking
- **📈 Results Tab**: Comprehensive performance visualization and metrics
- **🔍 Analysis Tab**: Detailed trading analysis and model insights
- **⚙️ Sidebar**: Agent selection, stock picker, and training configuration

#### Key Features:
- **Select RL Algorithm**: Choose between VAC and A3C agents
- **Pick Stocks**: Select from popular categories (Tech, Finance, Healthcare, etc.)
- **Configure Training**: Set parameters (window size, learning rate, episodes)
- **Monitor Progress**: Real-time training visualization
- **Analyze Results**: Interactive charts and performance metrics

### 3. Command Line Training

```bash
# Using uv (recommended)
uv run python main.py --mode train --symbol AAPL

# Train with A3C agent on Tesla
uv run python main.py --mode train --symbol TSLA --agent a3c --window-size 60 --episodes 1000

# Compare agents on the same stock
uv run python main.py --mode train --symbol GOOGL --agent vac --episodes 500
uv run python main.py --mode train --symbol GOOGL --agent a3c --episodes 500

# Advanced A3C training with custom settings and workers
uv run python main.py --mode train --symbol NVDA --agent a3c --workers 4 --train-period 3y --learning-rate 0.0005

# A3C with all CPU cores for maximum speed
uv run python main.py --mode train --symbol SPY --agent a3c --workers -1 --episodes 1000
```

### 4. Evaluate Trained Models

```bash
# Evaluate a VAC model
uv run python main.py --mode eval --model data/results_vac_AAPL_20240115_143022/best_model.pth --symbol AAPL --agent vac

# Evaluate an A3C model
uv run python main.py --mode eval --model data/results_a3c_TSLA_20240115_143022/best_model.pth --symbol TSLA --agent a3c
```

## 📋 Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `agent` | RL algorithm type | vac | vac, a3c |
| `workers` | A3C worker threads | auto | auto, -1 (all), 1-N |
| `symbol` | Stock ticker symbol | AAPL | Any valid ticker |
| `window_size` | Days of historical data | 30 | 10-120 |
| `train_period` | Data collection period | 2y | 1y, 2y, 3y, 5y |
| `learning_rate` | Neural network learning rate | 0.001 | 0.0001-0.01 |
| `episodes` | Training episodes | 500 | 100-2000 |

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

### 📈 Returns
- **Total Return**: Overall portfolio performance
- **Excess Return**: Performance vs. buy-and-hold strategy
- **Buy & Hold Return**: Baseline comparison metric

### 📉 Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Portfolio value fluctuation

### 🔄 Trading Metrics
- **Number of Trades**: Total buy/sell transactions
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Holding period analysis

## 🏛️ Algorithm Details

### Vanilla Actor-Critic (VAC)
The classic actor-critic implementation with:

1. **Shared Feature Extraction**: Common neural network layers for both actor and critic
2. **Policy Network (Actor)**: Outputs action probabilities using softmax
3. **Value Network (Critic)**: Estimates state values for advantage calculation
4. **Advantage Estimation**: Uses temporal difference for policy updates
5. **Synchronous Updates**: Updates after each episode completion

### A3C (Asynchronous Advantage Actor-Critic)
Advanced algorithm with true multi-threading:

1. **Multi-Threading**: Proper asynchronous training with multiple CPU worker threads
2. **Worker Customization**: Configure number of workers (auto, all cores, or custom count)
3. **Separate Networks**: Global and local networks for improved stability
4. **Entropy Regularization**: Encourages exploration through entropy bonus
5. **Gradient Clipping**: Prevents exploding gradients for stable training
6. **GAE (Generalized Advantage Estimation)**: More accurate advantage computation
7. **Enhanced Architecture**: Deeper networks with dropout for better generalization

### State Representation
Each state includes:
- **Technical Indicators**: RSI, Bollinger Bands, moving averages
- **Price Features**: Returns, high-low spreads, volume ratios
- **Portfolio State**: Current cash, position, and portfolio value

### Reward Function
- Primary reward based on portfolio value changes
- Normalized to encourage consistent performance
- Risk-adjusted to prevent excessive volatility

## 📁 Project Structure

```
StockPrediction/
├── main.py                    # Main application entry point
├── app.py                     # Streamlit web interface (modular)
├── pyproject.toml             # Project dependencies
├── README.md                  # This file
├── .streamlit/                # Streamlit configuration
│   ├── config.toml           # App configuration
│   └── secrets.toml          # Secrets template
│
├── components/                # 🧩 Modular UI Components
│   ├── __init__.py           # Component exports
│   ├── tools.py              # Utility functions & data processing
│   ├── sidebar.py            # Configuration sidebar
│   ├── overview.py           # Market overview tab
│   ├── training.py           # Training interface
│   ├── results.py            # Results visualization
│   └── analysis.py           # Detailed analysis
│
├── utils/                     # CLI utilities
│   ├── __init__.py           # Package exports
│   ├── parameters.py         # Parameter management
│   ├── training.py           # Training utilities
│   ├── evaluation.py         # Evaluation utilities
│   └── web.py                # Web interface utilities
│
├── trainers/                  # Training system
│   └── trainer.py            # Core training logic
│
├── agent/                     # RL agents
│   ├── __init__.py           # Agent exports
│   ├── base.py               # Base agent interface
│   ├── vac.py                # Vanilla Actor-Critic
│   └── a3c.py                # A3C implementation
│
├── envs/                      # Trading environment
│   └── stock_env.py          # Stock trading environment
│
└── data/                      # Training results (auto-generated)
    └── results_*/            # Results directories
        ├── best_model.pth
        ├── training_metrics.json
        ├── best_evaluation.json
        └── *.png             # Training plots
```

## 🎯 Usage Examples

### Example 1: Agent Comparison
```bash
# Compare VAC vs A3C on the same stock
uv run python main.py --mode train --symbol AAPL --agent vac --episodes 500
uv run python main.py --mode train --symbol AAPL --agent a3c --episodes 500
```

### Example 2: Technology Stock Analysis
```bash
# Compare A3C performance across tech stocks
uv run python main.py --mode train --symbol AAPL --agent a3c --episodes 500
uv run python main.py --mode train --symbol GOOGL --agent a3c --episodes 500
uv run python main.py --mode train --symbol MSFT --agent a3c --episodes 500
```

### Example 3: Different Training Windows
```bash
# Short-term A3C trading (10 days)
uv run python main.py --mode train --symbol TSLA --agent a3c --window-size 10 --episodes 750

# Long-term VAC analysis (90 days)
uv run python main.py --mode train --symbol TSLA --agent vac --window-size 90 --episodes 750
```

### Example 4: Market Comparison
```bash
# Individual stock vs ETF with different agents
uv run python main.py --mode train --symbol AAPL --agent vac
uv run python main.py --mode train --symbol SPY --agent a3c  # S&P 500 ETF
```

## 📈 Expected Results

Typical performance characteristics:

### 🎯 Well-Performing Scenarios
- **Trending Markets**: Algorithm excels in clear upward/downward trends
- **Volatile Stocks**: Benefits from frequent trading opportunities
- **Longer Training**: 500+ episodes generally show better convergence

### ⚠️ Challenging Scenarios
- **Sideways Markets**: Limited opportunities for excess returns
- **Highly Efficient Markets**: Difficult to beat buy-and-hold
- **Short Training**: May not learn optimal strategies

### 📊 Benchmark Comparisons
- Target: Beat buy-and-hold strategy
- Good Performance: 2-5% excess annual returns
- Excellent Performance: >5% excess returns with Sharpe ratio >1.0

## 🛠️ Advanced Usage

### Custom Training Loop
```python
from trainers.trainer import StockTrainer

# Initialize with A3C agent and custom parameters
trainer = StockTrainer(
    symbol="NVDA",
    window_size=45,
    train_period="3y",
    learning_rate=0.002,
    agent_type="a3c"
)

# Run training with custom evaluation
results = trainer.train(
    num_episodes=1000,
    eval_frequency=50,
    save_frequency=200
)

# Custom analysis
performance = trainer.evaluate_episode()
print(trainer.get_trading_summary(performance))
```

### Multi-Agent Ensemble
```python
from trainers.trainer import StockTrainer

# Train multiple agents for ensemble predictions
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
agents = ["vac", "a3c"]
models = {}

for symbol in symbols:
    for agent in agents:
        trainer = StockTrainer(symbol=symbol, agent_type=agent)
        results = trainer.train(num_episodes=500)
        models[f"{agent}_{symbol}"] = trainer.agent
```

### Component Development
```python
# Extending the modular components
from components import render_sidebar, render_training_tab

# Custom component integration
def custom_analysis_tab():
    st.header("Custom Analysis")
    # Your custom analysis logic here
    pass

# Add to main app
with st.tabs(["📊 Overview", "🤖 Training", "📈 Results", "🔍 Analysis", "🆕 Custom"]):
    # ... existing tabs ...
    with st.tab("🆕 Custom"):
        custom_analysis_tab()
```

## 🔧 Troubleshooting

### Common Issues

**1. Data Download Errors**
```
Error: No data found for symbol XYZ
```
- **Solution**: Verify ticker symbol is valid on Yahoo Finance
- **Fallback**: System automatically tries AAPL if custom symbol fails

**2. Training Convergence Issues**
```
Poor performance after many episodes
```
- **Solutions**:
  - Increase training episodes (try 1000+)
  - Adjust learning rate (try 0.0005 or 0.002)
  - Change window size (try 60 or 90 days)
  - Use more stable stocks (large-cap vs. penny stocks)

**3. Memory Issues**
```
CUDA out of memory / RAM issues
```
- **Solutions**:
  - Reduce batch size in training
  - Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`
  - Close other applications

### Performance Optimization

**For Better Results:**
- Use 2-3 years of training data
- Train for 500-1000 episodes
- Choose stocks with good volatility (not too stable, not too chaotic)
- Experiment with window sizes (30-60 days often work well)

**For Faster Training:**
- Reduce number of episodes
- Use smaller window sizes
- Train on CPU for small models

## 🚀 Deployment

### 🌐 Render.com Deployment

The application is configured for easy deployment on Render.com with proper port binding and environment configuration.

#### Prerequisites
1. **GitHub Repository**: Push your code to GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)

#### Deployment Steps

**1. Connect Repository**
- Go to Render Dashboard
- Click "New +" → "Web Service"
- Connect your GitHub repository

**2. Configure Service**
```
Name: stock-prediction-app
Environment: Python 3.11
Region: Choose closest to your users
Branch: main (or your default branch)
```

**3. Build & Deploy Settings**
```
Build Command: pip install uv && uv sync
Start Command: uv run streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

**4. Environment Variables**
```
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

**5. Deploy**
- Click "Create Web Service"
- Wait for build to complete (5-10 minutes)
- Your app will be available at `https://your-app-name.onrender.com`

#### 🐳 Docker Deployment

For containerized deployment using the provided Dockerfile:

```bash
# Build the image
docker build -t stock-prediction-app .

# Run the container
docker run -p 8501:8501 stock-prediction-app
```

#### 🔧 Local Production Testing

Test your deployment configuration locally:

```bash
# Run deployment preparation script
uv run python deploy.py

# Test with production settings
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

#### 📋 Deployment Checklist

- [ ] All required files present (`app.py`, `pyproject.toml`, `uv.lock`, etc.)
- [ ] Streamlit config updated for production (`address = "0.0.0.0"`)
- [ ] Environment variables configured
- [ ] Build and start commands set correctly
- [ ] Application tested locally with production settings

#### 🚨 Common Deployment Issues

**Port Binding Error**
```
No open ports detected on 0.0.0.0
```
- **Solution**: Ensure `STREAMLIT_SERVER_ADDRESS=0.0.0.0` in environment variables
- **Check**: `.streamlit/config.toml` has `address = "0.0.0.0"`

**Build Failures**
```
ModuleNotFoundError or dependency issues
```
- **Solution**: Use `pip install uv && uv sync` as build command
- **Check**: `pyproject.toml` and `uv.lock` are committed

**Memory Issues**
```
Application crashes or times out
```
- **Solution**: Upgrade to paid Render plan for more resources
- **Alternative**: Optimize code or reduce data processing

#### 🔄 Continuous Deployment

The application supports automatic deployments:
- Push to `main` branch triggers automatic deployment
- Use `render.yaml` for infrastructure as code
- Monitor deployment logs in Render dashboard

## 🤝 Contributing

Feel free to contribute by:
- Adding new RL algorithms (PPO, SAC, TD3, etc.)
- Implementing new technical indicators
- Improving the reward function
- Adding more sophisticated risk management
- Creating additional visualization tools
- Enhancing the agent comparison features
- **Extending the modular components** (new tabs, sidebar features, etc.)
- **Improving the component architecture** (better separation of concerns)
- **Adding new utility functions** to the tools module
- **Improving deployment configurations** and documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Yahoo Finance API** for providing free stock data
- **PyTorch** for deep learning framework
- **Streamlit** for the beautiful web interface
- **Reinforcement Learning Community** for algorithm insights

---

**Happy Trading! 📈💰**

*Remember: This is for educational and research purposes. Past performance does not guarantee future results. Always do your own research before making investment decisions.*
