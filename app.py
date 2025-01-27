import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Local modules for data handling, modeling, utilities, and plotting
from data_loader import fetch_data_yahoo, preprocess_data
from model import fit_arima_garch, forecast_arima_garch, auto_tune_arima_garch
from utils import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    compute_descriptive_stats,
    adf_test,
    ljung_box_test,
    arch_test
)
from plots import (
    price_plot, histogram_plot, qq_plot,
    acf_plot, pacf_plot, create_table_descriptive,
    create_table_forecast, residual_plot
)
import io
import base64
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# External stylesheet for Dash app styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # Expose the server variable for deployments

# Mapping of cryptocurrency names to their respective identifiers used in data fetching
CRYPTO_MAP = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Dogecoin (DOGE)': 'dogecoin',
    'Solana (SOL)': 'solana'
}

# Define the layout of the Dash app
app.layout = html.Div([
    # Title of the dashboard
    html.H1("ARIMA-GARCH Crypto Forecasting Dashboard", style={'textAlign': 'center'}),
    
    # ============== CONTROLS SECTION ==============
    html.Div([
        # 1) Error Distribution Selection
        html.Div([
            html.Label("Error Distribution:"),  # Label for the error distribution selection
            dcc.RadioItems(
                id='garch-distribution',  # Unique identifier for the component
                options=[
                    {'label': 'Normal', 'value': 'normal'},
                    {'label': "Student's t", 'value': 't'},
                    {'label': 'Skewed t', 'value': 'skewt'}
                ],
                value='normal',  # Default selection
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}  # Inline styling for labels
            )
        ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container

        # 2) Forecast Mode Selection
        html.Div([
            html.Label("Forecast Mode:"),  # Label for forecast mode selection
            dcc.RadioItems(
                id='forecast-mode',  # Unique identifier
                options=[
                    {'label': ' Backtest', 'value': 'backtest'},
                    {'label': ' Future', 'value': 'future'}
                ],
                value='backtest',  # Default selection
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}  # Inline styling
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container
        
        # 3) Parameter Mode Selection (Manual or Auto-Tune)
        html.Div([
            html.Label("Parameter Mode:"),  # Label for parameter mode selection
            dcc.RadioItems(
                id='param-mode',  # Unique identifier
                options=[
                    {'label': ' Manual', 'value': 'manual'},
                    {'label': ' Auto-Tune', 'value': 'auto'}
                ],
                value='manual',  # Default selection
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}  # Inline styling
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container
        
        # 4) Date Range Picker
        html.Div([
            html.Label("Select Date Range:"),  # Label for date range selection
            dcc.DatePickerRange(
                id='date-range',  # Unique identifier
                min_date_allowed=datetime(2015, 1, 1),  # Earliest selectable date
                max_date_allowed=datetime.today(),  # Latest selectable date
                start_date=datetime(2021, 1, 1),  # Default start date
                end_date=datetime.today(),  # Default end date
                display_format='YYYY-MM-DD'  # Format for displaying dates
            )
        ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container

        # 5) Cryptocurrency Dropdown Selector
        html.Div([
            html.Label("Select Cryptocurrency:"),  # Label for cryptocurrency selection
            dcc.Dropdown(
                id='crypto-dropdown',  # Unique identifier
                options=[{'label': k, 'value': v} for k, v in CRYPTO_MAP.items()],  # Options generated from CRYPTO_MAP
                value='bitcoin',  # Default selection
                clearable=False  # Prevents clearing the selection
            )
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '15px'}),  # Styling for the container
        
        # 6) ARIMA Parameters Input (p, d, q)
        html.Div([
            html.Label("ARIMA Parameters:"),  # Label for ARIMA parameters
            html.Div([
                dcc.Input(id='arima-p', type='number', value=1, min=0, 
                          style={'width': '60px', 'marginRight': '10px'}),  # Input for ARIMA p
                dcc.Input(id='arima-d', type='number', value=0, min=0,
                          style={'width': '60px', 'marginRight': '10px'}),  # Input for ARIMA d
                dcc.Input(id='arima-q', type='number', value=1, min=0,
                          style={'width': '60px'})  # Input for ARIMA q
            ])
        ], style={'width': '15%', 'display': 'inline-block'}),  # Styling for the container
        
        # 7) GARCH Parameters Input (p, q)
        html.Div([
            html.Label("GARCH Parameters:"),  # Label for GARCH parameters
            html.Div([
                dcc.Input(id='garch-p', type='number', value=1, min=0,
                          style={'width': '60px', 'marginRight': '10px'}),  # Input for GARCH p
                dcc.Input(id='garch-q', type='number', value=1, min=0,
                          style={'width': '60px'})  # Input for GARCH q
            ])
        ], style={'width': '15%', 'display': 'inline-block'}),  # Styling for the container
        
        # 8) Forecast Horizon Input (Number of Days)
        html.Div([
            html.Label("Forecast Days:"),  # Label for forecast horizon
            dcc.Input(id='forecast-horizon', type='number', value=30, min=1,
                      style={'width': '100px'})  # Input for forecast horizon
        ], style={'width': '10%', 'display': 'inline-block'}),  # Styling for the container
    ], style={'padding': '20px', 'borderBottom': '1px solid #ddd'}),  # Styling for the controls section
    
    # ============== ACTION SECTION ==============
    html.Div([
        # Button to run the analysis
        html.Button("Run Analysis", id='run-button', n_clicks=0,
                    style={'backgroundColor': '#4CAF50', 'color': 'white', 
                           'padding': '10px 20px', 'borderRadius': '5px'}),  # Styling for the button
        # Button to refresh the data
        html.Button("Refresh Data", id='refresh-button', n_clicks=0,
                    style={'backgroundColor': '#2196F3', 'color': 'white',
                           'padding': '10px 20px', 'marginLeft':'10px',
                           'borderRadius': '5px'}),  # Styling for the button
        # Div to display status messages
        html.Div(id='status-message', style={'color': 'red', 'marginLeft': '20px'})  # Styling for status message
    ], style={'padding': '20px'}),  # Styling for the action section

    # ============== MAIN CONTENT ==============
    html.Div([
        # Left side: Plots
        html.Div([
            # Price Plot
            dcc.Graph(id='price-plot', style={'height': '400px'}),  # Graph component for price plot
            
            # Histogram and Q-Q Plot Side by Side
            html.Div([
                html.Div([dcc.Graph(id='hist-plot')], 
                         style={'width': '50%', 'display': 'inline-block'}),  # Histogram plot
                html.Div([dcc.Graph(id='qq-plot')], 
                         style={'width': '50%', 'display': 'inline-block'})  # Q-Q plot
            ]),
            
            # ACF and PACF Plots Side by Side
            html.Div([
                html.Div([dcc.Graph(id='acf-plot')], 
                         style={'width': '50%', 'display': 'inline-block'}),  # ACF plot
                html.Div([dcc.Graph(id='pacf-plot')], 
                         style={'width': '50%', 'display': 'inline-block'})  # PACF plot
            ]),

            # Residual Plot
            dcc.Graph(id='resid-plot', style={'height': '300px'}),  # Residuals plot

        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),  # Styling for the left side
        
        # Right side: Tables & Diagnostics
        html.Div([
            # Performance Metrics Section
            html.H3("Performance Metrics"),  # Header for performance metrics
            html.Div(id='stats-table', style={'marginBottom': '20px'}),  # Container for stats table

            # Forecast Prices Section
            html.H3("Forecast Prices"),  # Header for forecast prices
            html.Div(id='forecast-table'),  # Container for forecast table

            # Diagnostics Section
            html.H3("Diagnostics"),  # Header for diagnostics
            html.Div(id='diagnostics-summary', style={'whiteSpace': 'pre-wrap'})  # Container for diagnostics summary
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 
                  'padding': '20px', 'backgroundColor': '#f9f9f9'})  # Styling for the right side
    ])
])

# ============== CALLBACKS ==============

@app.callback(
    [
        Output('arima-p', 'disabled'),  # Output to disable ARIMA p input
        Output('arima-d', 'disabled'),  # Output to disable ARIMA d input
        Output('arima-q', 'disabled'),  # Output to disable ARIMA q input
        Output('garch-p', 'disabled'),  # Output to disable GARCH p input
        Output('garch-q', 'disabled')   # Output to disable GARCH q input
    ],
    [Input('param-mode', 'value')]  # Input: value of parameter mode radio items
)
def toggle_inputs(mode):
    disabled = (mode == 'auto')
    return [disabled, disabled, disabled, disabled, disabled]


@app.callback(
    [
        Output('status-message', 'children'),
        Output('price-plot', 'figure'),
        Output('hist-plot', 'figure'),
        Output('qq-plot', 'figure'),
        Output('acf-plot', 'figure'),
        Output('pacf-plot', 'figure'),
        Output('resid-plot', 'figure'),
        Output('stats-table', 'children'),
        Output('forecast-table', 'children'),
        Output('diagnostics-summary', 'children')
    ],
    [
        Input('run-button', 'n_clicks'),
        Input('refresh-button', 'n_clicks'),
        Input('garch-distribution', 'value')
    ],
    [
        State('date-range', 'start_date'),
        State('date-range', 'end_date'),
        State('crypto-dropdown', 'value'),
        State('param-mode', 'value'),
        State('forecast-mode', 'value'),
        State('arima-p', 'value'),
        State('arima-d', 'value'),
        State('arima-q', 'value'),
        State('garch-p', 'value'),
        State('garch-q', 'value'),
        State('forecast-horizon', 'value')
    ]
)
def update_all_components(run_clicks,
                          refresh_clicks,
                          garch_dist,
                          start_date,
                          end_date,
                          coin_id,
                          param_mode,
                          forecast_mode,
                          p, d, q,
                          garch_p, garch_q,
                          horizon):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        # 1) Data Fetching
        if start_date:
            start_date = start_date.split('T')[0]
        if end_date:
            end_date = end_date.split('T')[0]

        raw_df = fetch_data_yahoo(coin_id, start=start_date, end=end_date)
        status_msg = "Data loaded."
        if trigger_id == 'refresh-button':
            status_msg = "Data refreshed from Yahoo Finance."

        # 2) Data Preprocessing
        processed_df = preprocess_data(raw_df)
        if len(processed_df) < 30:
            raise ValueError("Insufficient data (minimum 30 days required).")

        # 3) Stationarity Check using ADF Test
        adf_result = adf_test(processed_df['log_return'])
        adf_text = (f"ADF p-value={adf_result['p_value']:.4f}. "
                    f"Stationary? {adf_result['is_stationary']}\n")
        differenced = False
        if not adf_result['is_stationary']:
            processed_df['log_return'] = processed_df['log_return'].diff().dropna()
            differenced = True
            adf_text += " => Non-stationary. Applied 1st difference.\n"

        # 4) Splitting Data for Backtest or Future Forecast
        if forecast_mode == 'backtest':
            split_index = len(processed_df) - horizon
            if split_index < horizon:
                raise ValueError(f"Insufficient data for a {horizon}-day backtest. Need at least {2 * horizon} days.")
            train_df = processed_df.iloc[:split_index]
            test_df = processed_df.iloc[split_index:]
            forecast_dates = test_df['date'].values
        else:
            train_df = processed_df
            test_df = pd.DataFrame()
            last_date = pd.to_datetime(train_df['date'].iloc[-1])
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

        # 5) ARIMA/GARCH Parameter Selection
        if param_mode == 'auto':
            best_params = auto_tune_arima_garch(train_df['log_return'].dropna())
            p, d, q = best_params['arima']
            garch_p, garch_q = best_params['garch']
            param_status = f"Auto: ARIMA({p},{d},{q}), GARCH({garch_p},{garch_q})"
        else:
            param_status = f"Manual: ARIMA({p},{d},{q}), GARCH({garch_p},{garch_q})"

        # 6) Model Fitting with ARIMA and GARCH
        try:
            arima_model, garch_model, scale = fit_arima_garch(
                train_df['log_return'].dropna(),
                arima_order=(p, d, q),
                garch_order=(garch_p, garch_q),
                dist=garch_dist,
                rescale_data=True,
                scale_factor=1000
            )
        except Exception as e:
            raise ValueError(f"Model fitting error: {e}")

        # 7) Residual Analysis
        final_resid_series = pd.Series(garch_model.std_resid, index=train_df.index[-len(garch_model.std_resid):])

        # Perform Ljung-# Perform Ljung-Box Test for White Noise
        lb_resid = final_resid_series.copy()
        lb_result = ljung_box_test(lb_resid)

        # Perform Engle's ARCH Test for Heteroskedasticity
        arch_resid = final_resid_series.copy()
        arch_result = arch_test(arch_resid, lags=12)

        adf_check = "✅" if adf_result['is_stationary'] else "❌"
        lb_check = "✅" if lb_result['is_white_noise'] else "❌"
        arch_check = "✅" if not arch_result['heteroskedastic'] else "❌"

        diag_message = (
            f"ADF p-value={adf_result['p_value']:.4f}. Stationary? {adf_result['is_stationary']} {adf_check}\n"
            f"{'Differenced log_return.\n' if differenced else ''}"
            f"Ljung-Box Q p-value={lb_result['lb_pvalue']:.4f}. White Noise? {lb_result['is_white_noise']} {lb_check}\n"
            f"Engle's ARCH p-value={arch_result['arch_pvalue']:.4f}. Heteroskedastic? {arch_result['heteroskedastic']} {arch_check}\n"
        )

        # 8) Forecast Generation
        forecast_out = forecast_arima_garch(arima_model, garch_model, horizon, scale)
        if forecast_mode == 'backtest':
            last_train_price = train_df['price'].iloc[-1]
            reconstructed_price = last_train_price * np.exp(forecast_out['mean_return'].cumsum())
        else:
            last_price = train_df['price'].iloc[-1]
            reconstructed_price = last_price * np.exp(forecast_out['mean_return']).cumprod()

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast_price': reconstructed_price
        })

        # 9) Performance Metrics Calculation
        metrics = {}
        if forecast_mode == 'backtest' and not test_df.empty:
            merged = pd.merge(test_df, forecast_df, on='date', how='inner')
            if not merged.empty:
                mae_ = mean_absolute_error(merged['price'], merged['forecast_price'])
                rmse_ = root_mean_squared_error(merged['price'], merged['forecast_price'])
                mape_ = mean_absolute_percentage_error(merged['price'], merged['forecast_price'])
                metrics = {'mae': mae_, 'rmse': rmse_, 'mape': mape_}
                metric_text = f"MAE={mae_:.2f}, RMSE={rmse_:.2f}, MAPE={mape_:.2f}%"
            else:
                metric_text = "Error: No overlapping dates for backtesting."
        else:
            metric_text = f"Forecast {horizon} days (no backtest)"

        stats = {
            **compute_descriptive_stats(processed_df),
            'model_aic': arima_model.aic + garch_model.aic,
            'model_bic': arima_model.bic + garch_model.bic,
            **metrics
        }

        # 10) Plot Generation

        # Price Plot
        price_fig = price_plot(train_df, forecast_df, forecast_mode)

        # Histogram Plot
        hist_fig = histogram_plot(processed_df, garch_dist)

        # Q-Q Plot
        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)
        sm.qqplot(processed_df['log_return'].dropna(), line='45', fit=True, ax=ax)
        ax.set_title('Q-Q Plot of Log Returns')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')

        qq_fig = go.Figure()
        qq_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",
                x=0, y=1, xref="paper", yref="paper",
                sizex=1, sizey=1, layer="below"
            )
        )
        qq_fig.update_layout(
            width=700, height=500,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=40, b=40),
            template='plotly_white'
        )

        # ACF Plot
        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(1, 1, 1)
        plot_acf(processed_df['log_return'].dropna(), lags=20, alpha=0.05, ax=ax)
        buf_acf = io.BytesIO()
        fig.savefig(buf_acf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf_acf.seek(0)
        encoded_image_acf = base64.b64encode(buf_acf.read()).decode('utf-8')

        acf_fig = go.Figure()
        acf_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image_acf}",
                x=0, y=1, xref="paper", yref="paper",
                sizex=1, sizey=1, layer="below"
            )
        )
        acf_fig.update_layout(
            width=700, height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=40, b=40),
            template='plotly_white',
            title='ACF'
        )

        # PACF Plot
        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(1, 1, 1)
        plot_pacf(processed_df['log_return'].dropna(), lags=20, alpha=0.05, method='ols', ax=ax)
        buf_pacf = io.BytesIO()
        fig.savefig(buf_pacf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf_pacf.seek(0)
        encoded_image_pacf = base64.b64encode(buf_pacf.read()).decode('utf-8')

        pacf_fig = go.Figure()
        pacf_fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image_pacf}",
                x=0, y=1, xref="paper", yref="paper",
                sizex=1, sizey=1, layer="below"
            )
        )
        pacf_fig.update_layout(
            width=700, height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(t=40, b=40),
            template='plotly_white',
            title='PACF'
        )

        # Residual Plot
        resid_fig = residual_plot(final_resid_series)

        # Descriptive Statistics Table
        stats_table = create_table_descriptive(stats)

        # Forecasted Prices Table
        forecast_table = create_table_forecast(forecast_df)

        # Compile the full status message
        status_full = f"{status_msg} | {param_status} | {metric_text}"

        return (
            status_full,
            price_fig,
            hist_fig,
            qq_fig,
            acf_fig,
            pacf_fig,
            resid_fig,
            stats_table,
            forecast_table,
            diag_message
        )

    except ValueError as ve:
        return (
            f"Validation Error: {str(ve)}",
            go.Figure(), go.Figure(), go.Figure(),
            go.Figure(), go.Figure(), go.Figure(),
            [], [], ""
        )
    except Exception as e:
        return (
            f"Unexpected Error: {str(e)}",
            go.Figure(), go.Figure(), go.Figure(),
            go.Figure(), go.Figure(), go.Figure(),
            [], [], ""
        )

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False) 