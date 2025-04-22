import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt

##########INPUT##########
Asset1_tickerz = "BZ=F"
Asset2_tickerz = "CL=F"
startz = "2015-04-22"
endz = "2025-04-22"
intervalz = "1D"
transaction_costz = 0.005
initial_capitalz = 1000
rolling_windowz = 60
leveragez = 2
soglia_z_scorez = 1.5
in_sample_yearsz = 2
out_sample_yearsz = 1
stop_lossz = 0.025
take_profitz = 2
##########################

st.set_page_config(page_title="Pair Trading Backtest", layout="wide")
st.title("Pair Trading Backtest")

with st.form(key="params_form"):
    st.subheader("Parameters of the strategy")
    col1, col2 = st.columns(2)
    with col1:
        Asset1_ticker = st.text_input("Ticker Asset 1", value= Asset1_tickerz)
        start = st.date_input("Start Date", pd.to_datetime(startz))
        transaction_cost = st.number_input("Transaction Cost (%)", value=transaction_costz)
        leverage = st.number_input("Leverage", value=leveragez)
        rolling_window = st.number_input("Rolling Window", value=rolling_windowz)
        stop_loss = st.number_input("Stop Loss (%)", value=stop_lossz)
    with col2:
        Asset2_ticker = st.text_input("Ticker Asset 2", value=Asset2_tickerz)
        end = st.date_input("End Date", pd.to_datetime(endz))
        interval = st.selectbox("Frequency", ["1d", "1h", "1wk"])
        initial_capital = st.number_input("Starting Capital", value=initial_capitalz)
        soglia_z_score = st.number_input("Z-score Threshold", value=soglia_z_scorez)
        take_profit = st.number_input("Take Profit (%)", value=take_profitz)

    in_sample_years = st.number_input("In-Sample Years", value=in_sample_yearsz)
    out_sample_years = st.number_input("Out-of-Sample Years", value=out_sample_yearsz)

    submitted = st.form_submit_button("Backtest")

if submitted:
    st.write("""---""")
    st.write("Loading data...")


    def get_data(ticker, start, end, interval=interval):
        data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
        data = data.ffill().dropna()
        data['Returns'] = data['Close'].pct_change()
        data = data.dropna()
        data = data[['Close', 'Returns']]
        return data


    def Metrics_Portfolio(prices: pd.DataFrame, label: str = "Asset"):
        returns = prices.pct_change()
        mean_return = returns.mean().iloc[0]
        annualized_return = mean_return * 252

        downside_returns = returns[returns < 0]
        dsd = downside_returns.std().iloc[0] * (252 ** 0.5)

        sd = returns.std().iloc[0] * (252 ** 0.5)

        infra = annualized_return / sd
        infra_down = annualized_return / dsd

        cum_ret = (1 + returns).cumprod()
        cum_ret2 = cum_ret.iloc[-1, 0] - 1
        max_cum = cum_ret.cummax()

        drawdown = (max_cum - cum_ret) / max_cum
        Max_draw = drawdown.max().iloc[0]

        cal = annualized_return / Max_draw

        plt.plot(cum_ret, label="Cumulative Return")
        plt.yscale("log")
        plt.plot(max_cum, label="Maximum Cumulative Return")
        plt.title("Portfolio")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Time")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        metrics_dict = {
            "Cumulative Return": f"{cum_ret2:.2%}",
            "Mean Return": f"{mean_return:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Standard Deviation": f"{sd:.2%}",
            "Information Ratio": f"{infra:.2f}",
            "Downside Deviation": f"{dsd:.2%}",
            "Sortino Ratio": f"{infra_down:.2f}",
            "Max Drawdown": f"{Max_draw:.2%}",
            "Calmar Ratio": f"{cal:.2f}",
        }

        metrics_df = pd.DataFrame(metrics_dict.items(), columns=["Metric", "Value"])

        st.subheader(f"📈 {label} - Performance Metrics")
        st.table(metrics_df)

    def backtest(data, in_sample_years, out_sample_years, initial_capital, leverage,
                 transaction_cost, soglia_z_score, stop_loss, take_profit,
                 start_date, end_date, rolling_window):
        z_score_columns = [
            "Rolling Z Score Spread",
            "Rolling Z Score Ratio"
        ]

        all_results = {}

        def build_equity_curve(trades_df, initial_capital, start_date, end_date):
            if trades_df.empty:
                full_index = pd.date_range(start=start_date, end=end_date)
                return pd.Series(initial_capital, index=full_index)

            net_returns = trades_df["net_return"]
            net_returns.index = pd.to_datetime(trades_df.index)
            full_index = pd.date_range(start=start, end=end)
            daily_returns = pd.Series(0.0, index=full_index, dtype=float)
            daily_returns.loc[net_returns.index] = net_returns.values
            equity_curve = (1 + daily_returns).cumprod() * initial_capital
            return equity_curve

        for z_col in z_score_columns:
            capital = initial_capital
            results = []
            trade_log = []
            start_date = data.index.min()
            end_date = data.index.max()
            current_start = start_date

            while True:
                in_sample_end = current_start + pd.DateOffset(years=in_sample_years)
                out_sample_end = in_sample_end + pd.DateOffset(years=out_sample_years)

                if out_sample_end > end_date:
                    break

                in_sample_data = data[(data.index >= current_start) & (data.index < in_sample_end)].copy()
                out_sample_data = data[(data.index >= in_sample_end) & (data.index < out_sample_end)].copy()

                if z_col == "Rolling Z Score Spread":
                    out_sample_data["spread"] = out_sample_data["Close_A1"] - out_sample_data["Close_A2"]
                    rolling_mean = out_sample_data["spread"].rolling(window=rolling_window).mean()
                    rolling_std = out_sample_data["spread"].rolling(window=rolling_window).std()
                    out_sample_data["z_score"] = (out_sample_data["spread"] - rolling_mean) / rolling_std

                elif z_col == "Rolling Z Score Ratio":
                    out_sample_data["ratio"] = out_sample_data["Close_A1"] / out_sample_data["Close_A2"]
                    rolling_mean = out_sample_data["ratio"].rolling(window=rolling_window).mean()
                    rolling_std = out_sample_data["ratio"].rolling(window=rolling_window).std()
                    out_sample_data["z_score"] = (out_sample_data["ratio"] - rolling_mean) / rolling_std

                position = 0
                entry_price_A1 = None
                entry_price_A2 = None
                daily_capital = capital

                for i in range(1, len(out_sample_data)):
                    z = out_sample_data["z_score"].iloc[i]
                    curr_A1 = out_sample_data["Close_A1"].iloc[i]
                    curr_A2 = out_sample_data["Close_A2"].iloc[i]
                    prev_A1 = out_sample_data["Close_A1"].iloc[i - 1]
                    prev_A2 = out_sample_data["Close_A2"].iloc[i - 1]

                    if position == 0:
                        if z > soglia_z_score:
                            position = -1
                            entry_price_A1 = curr_A1
                            entry_price_A2 = curr_A2
                        elif z < -soglia_z_score:
                            position = 1
                            entry_price_A1 = curr_A1
                            entry_price_A2 = curr_A2

                    elif position != 0:
                        ret_A1 = (curr_A1 - entry_price_A1) / entry_price_A1
                        ret_A2 = (curr_A2 - entry_price_A2) / entry_price_A2
                        gross_return = position * (ret_A1 - ret_A2)
                        net_return = gross_return * leverage - transaction_cost

                        if net_return >= take_profit * leverage or net_return <= -stop_loss * leverage or (
                                (position == 1 and z >= 0) or (position == -1 and z <= 0)
                        ):
                            trade_log.append({
                                "entry_date": out_sample_data.index[i - 1],
                                "exit_date": out_sample_data.index[i],
                                "type": "long_spread" if position == 1 else "short_spread",
                                "entry_price_A1": entry_price_A1,
                                "entry_price_A2": entry_price_A2,
                                "exit_price_A1": curr_A1,
                                "exit_price_A2": curr_A2,
                                "gross_return": gross_return,
                                "net_return": net_return,
                                "z_score_entry": out_sample_data["z_score"].iloc[i - 1],
                                "z_score_exit": z
                            })

                            daily_capital *= (1 + net_return)
                            position = 0
                            entry_price_A1 = None
                            entry_price_A2 = None

                pnl = daily_capital - capital
                capital = daily_capital

                results.append({
                    "start_in": current_start,
                    "end_in": in_sample_end,
                    "start_out": in_sample_end,
                    "end_out": out_sample_end,
                    "pnl": pnl,
                    "capital_after": capital,
                    "num_days": len(out_sample_data),
                })

                current_start = current_start + pd.DateOffset(years=out_sample_years)

            results_df = pd.DataFrame(results)
            results_df.index = pd.to_datetime(results_df["start_out"])
            results_df.columns.name = "Metric"
            results_df.index.name = "Date"
            results_df = results_df.sort_index()
            results_df = results_df.rename(columns={
                "start_in": "In-Sample Start",
                "end_in": "In-Sample End",
                "start_out": "Out-Sample Start",
                "end_out": "Out-Sample End",
                "pnl": "PnL",
                "capital_after": "Capital (€)",
                "num_days": "Days"
            })

            trades_df = pd.DataFrame(trade_log)
            if not trades_df.empty:
                trades_df.index = pd.to_datetime(trades_df["exit_date"])
                trades_df.columns.name = "Trade Info"
                trades_df.index.name = "Date"
                trades_df = trades_df.sort_index()

            equity_curve = build_equity_curve(trades_df, initial_capital, start, end)

            all_results[z_col] = {
                "results": results_df,
                "trades": trades_df,
                "equity_curve": equity_curve
            }

        return all_results


    Asset1 = get_data(Asset1_ticker, start, end, interval).add_suffix('_A1')
    Asset2 = get_data(Asset2_ticker, start, end, interval).add_suffix('_A2')
    data = pd.concat([Asset1, Asset2], axis=1)
    data.columns = data.columns.get_level_values(0)
    data = data.ffill().dropna()
    reg = sm.OLS(data["Close_A2"], data["Close_A1"]).fit()
    b = reg.params["Close_A1"]
    # rolling_reg = sm.OLS(data["Close_A2"],data["Close_A1"], rolling_type = rolling_window).fit()
    # b_rolling = rolling_reg.params["Close_A1"]
    data["ratio_prezzi"] = data['Close_A1'] / data['Close_A2']
    data["spread"] = data['Close_A1'] - b * data['Close_A2']
    # data["rolling_spread"] = data['Close_A1'] - b_rolling*data['Close_A2']

    Correlation = data[['Returns_A1', 'Returns_A2']].corr()
    data["rolling_correlation"] = data['Returns_A1'].rolling(window=rolling_window).corr(data['Returns_A2']).dropna()

    score, p_value, _ = coint(data['Close_A1'], data['Close_A2'])

    adf_result_spread = adfuller(data["spread"])
    adf_result_ratio = adfuller(data["ratio_prezzi"])
    # rolling_adf_spread = rolling_adf(data["spread"], rolling_window)
    # rolling_adf_ratio = rolling_adf(data["ratio_prezzi"], rolling_window)

    spread_mean = data["spread"].mean()
    spread_std = data["spread"].std()
    data["z_score_spread"] = (data["spread"] - spread_mean) / spread_std

    long_signal_1 = data["z_score_spread"] < -soglia_z_score
    short_signal_1 = data["z_score_spread"] > soglia_z_score

    ratio_mean = data["ratio_prezzi"].mean()
    ratio_std = data["ratio_prezzi"].std()
    data["z_score_ratio"] = (data["ratio_prezzi"] - ratio_mean) / ratio_std

    long_signal_2 = data["z_score_ratio"] < -soglia_z_score
    short_signal_2 = data["z_score_ratio"] > soglia_z_score

    data["rolling_mean_spread"] = data["spread"].rolling(window=rolling_window).mean()
    data["rolling_std_spread"] = data["spread"].rolling(window=rolling_window).std()
    data["Rolling Z Score Spread"] = (data["spread"] - data["rolling_mean_spread"]) / data["rolling_std_spread"]

    long_signal_3 = data["Rolling Z Score Spread"] < -soglia_z_score
    short_signal_3 = data["Rolling Z Score Spread"] > soglia_z_score

    data["rolling_mean_ratio"] = data["ratio_prezzi"].rolling(window=rolling_window).mean()
    data["rolling_std_ratio"] = data["ratio_prezzi"].rolling(window=rolling_window).std()
    data["Rolling Z Score Ratio"] = (data["ratio_prezzi"] - data["rolling_mean_ratio"]) / data["rolling_std_ratio"]

    long_signal_4 = data["Rolling Z Score Ratio"] < -soglia_z_score
    short_signal_4 = data["Rolling Z Score Ratio"] > soglia_z_score

    Bt = backtest(data, in_sample_years, out_sample_years, initial_capital, leverage, transaction_cost, soglia_z_score,
                  stop_loss, take_profit, start, end, rolling_window)

    st.subheader("Cointegration and Stationarity test:")

    st.markdown(f"""
    - **Cointegration test**: `{p_value:.5f}` → {'✅ Cointegrated' if p_value < 0.05 else '❌ Not cointegrated'}
    - **ADF Spread**: `{adf_result_spread[1]:.5f}` → {'✅ Stationary' if adf_result_spread[1] < 0.05 else '❌ Not stationary'}
    - **ADF Ratio**: `{adf_result_ratio[1]:.5f}` → {'✅ Stationary' if adf_result_ratio[1] < 0.05 else '❌ Not stationary'}
    """)

    for strategy_name, results in Bt.items():
        print(f"\n--- Metrics for: {strategy_name} ---\n")
        equity = results["equity_curve"].to_frame(name="Portfolio")
        Metrics_Portfolio(equity, label=strategy_name)

    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Close_A1'], label=Asset1_ticker, color='blue')
    plt.plot(data.index, data['Close_A2'], label=Asset2_ticker, color='red')
    plt.title("Prices of " + str(Asset1_ticker) + " and " + str(Asset2_ticker))
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(14, 6))
    plt.plot(data["rolling_correlation"], label="Rolling Correlation (" + str(rolling_window) + ")")
    plt.title("Rolling Correlation")
    plt.ylabel("Correlation")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("📊 Static Correlation")
    st.dataframe(Correlation)

    plt.figure(figsize=(14, 6))
    plt.plot(data["spread"], label='Spread')
    plt.axhline(spread_mean, color='blue', linestyle='--', label='Spread Mean')
    plt.axhline(spread_mean + soglia_z_score * spread_std, color='green', linestyle='--',
                label='Upper Threshold (+' + str(soglia_z_score) + 'σ)')
    plt.axhline(spread_mean - soglia_z_score * spread_std, color='red', linestyle='--',
                label='Lower Threshold (-' + str(soglia_z_score) + 'σ)')
    plt.legend()
    plt.title("Spread and Trading Signals")
    plt.ylabel("Spread")
    plt.xlabel("Time")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(14, 6))
    plt.plot(data["ratio_prezzi"], label='Ratio Price')
    plt.axhline(ratio_mean, color='blue', linestyle='--', label='Ratio Mean')
    plt.axhline(ratio_mean + soglia_z_score * ratio_std, color='green', linestyle='--',
                label='Upper Threshold (+' + str(soglia_z_score) + 'σ)')
    plt.axhline(ratio_mean - soglia_z_score * ratio_std, color='red', linestyle='--',
                label='Lower Threshold (-' + str(soglia_z_score) + 'σ)')
    plt.legend()
    plt.title("Ratio and Trading Signals")
    plt.ylabel("Ratio")
    plt.xlabel("Time")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(14, 6))
    plt.plot(data["Rolling Z Score Spread"], label='Z-Score Spread Rolling (60g)')
    plt.axhline(soglia_z_score, color='green', linestyle='--', alpha=0.5)
    plt.axhline(-soglia_z_score, color='red', linestyle='--', alpha=0.5)
    plt.title("Static Z-Score Spread vs Rolling (60 giorni)")
    plt.xlabel("Date")
    plt.ylabel("Z-Score Spread")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(14, 6))
    plt.plot(data["Rolling Z Score Ratio"], label='Z-Score Ratio Rolling (60g)')
    plt.axhline(soglia_z_score, color='green', linestyle='--', alpha=0.5)
    plt.axhline(-soglia_z_score, color='red', linestyle='--', alpha=0.5)
    plt.title("Static Z-Score Ratio vs Rolling (60 giorni)")
    plt.xlabel("Date")
    plt.ylabel("Z-Score Ratio")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
