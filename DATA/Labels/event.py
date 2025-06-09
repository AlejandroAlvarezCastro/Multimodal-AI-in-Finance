from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pathlib import Path
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

from visualizer import EventVisualizer

@dataclass
class Event:
    event_date: str
    ticker: str
    quarter: str
    year: str
    market_ticker: str = 'SPGI'
    folder_path: str = 'companies_closes'
    t1_offset: int = -7
    t2_offset: int = 7
    plot_base_path: str = 'plots'

    alpha: float = field(init=False)
    beta: float = field(init=False)
    df_ar: pd.DataFrame = field(init=False)
    car: float = field(init=False)

    def __post_init__(self):
        self._calculate_event_window()
        self._estimate_market_model()
        self._calculate_abnormal_returns()
        self._calculate_car()

    def _calculate_event_window(self):
        t = datetime.strptime(self.event_date, "%Y-%m-%d")
        self.estimation_start = (t - timedelta(days=80)).strftime("%Y-%m-%d")
        self.estimation_end = (t - timedelta(days=27)).strftime("%Y-%m-%d")
        self.event_start = (t - timedelta(days=7)).strftime("%Y-%m-%d")
        self.event_end = (t + timedelta(days=7)).strftime("%Y-%m-%d")

    def _estimate_market_model(self):
        df_accion = self._load_returns(self.ticker)
        df_mercado = self._load_returns(self.market_ticker)

        df_merged = pd.merge(df_accion, df_mercado, on='Date', suffixes=('_accion', '_mercado'))
        df_ventana = df_merged[(df_merged['Date'] >= self.estimation_start) & (df_merged['Date'] <= self.estimation_end)].dropna()

        X = sm.add_constant(df_ventana['Return_mercado'])
        y = df_ventana['Return_accion']
        modelo = sm.OLS(y, X).fit()
        self.alpha, self.beta = modelo.params

    def _calculate_abnormal_returns(self):
        df_accion = self._load_returns(self.ticker)
        df_mercado = self._load_returns(self.market_ticker)
        df_merged = pd.merge(df_accion, df_mercado, on='Date', suffixes=('_accion', '_mercado'))

        df_evento = df_merged[(df_merged['Date'] >= self.event_start) & (df_merged['Date'] <= self.event_end)].dropna()
        df_evento['Return_esperado'] = self.alpha + self.beta * df_evento['Return_mercado']
        df_evento['AR'] = df_evento['Return_accion'] - df_evento['Return_esperado']
        self.df_ar = df_evento[['Date', 'Return_accion', 'Return_mercado', 'Return_esperado', 'AR']]

    def _calculate_car(self):
        df = self.df_ar.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        event_day = pd.to_datetime(self.event_date)
        df['t'] = (df['Date'] - event_day).dt.days
        df['CAR'] = df['AR'].cumsum()
        df_sub = df[(df['t'] >= self.t1_offset) & (df['t'] <= self.t2_offset)]
        self.car = df_sub['AR'].sum()
        self.df_ar = df


    def generate_and_save_plots(self):
        output_dir = Path(self.plot_base_path) / self.year / self.quarter
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = EventVisualizer(self)

        visualizer.plot_ar_car(save_path=output_dir / f"{self.ticker}_ar_car.png")
        visualizer.plot_periodos_evento2(save_path=output_dir / f"{self.ticker}_timeline.png")

    def _load_returns(self, ticker):
        path = os.path.join(self.folder_path, f'{ticker}_historico.csv')
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df[['Date', 'Return']].dropna()

    def summary_dict(self):
        return {
            'ticker': self.ticker,
            'event_date': self.event_date,
            'estimation_start': self.estimation_start,
            'estimation_end': self.estimation_end,
            'event_start': self.event_start,
            'event_end': self.event_end,
            'alpha': round(self.alpha, 6),
            'beta': round(self.beta, 6),
            'car': round(self.car * 100, 4),  # en porcentaje
            't1_offset': self.t1_offset,
            't2_offset': self.t2_offset,
            'quarter': self.quarter,
            'year': self.year,
        }
