
import matplotlib.pyplot as plt
import pandas as pd

class EventVisualizer:
    def __init__(self, event):
        self.event = event

    def plot_ar_car(self, save_path=None):
        df = self.event.df_ar.copy()
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(df['Date'], df['AR'], color='skyblue', label='AR diario')
        ax1.set_ylabel('Retorno Anormal Diario')
        ax1.axhline(0, color='gray', linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(df['Date'], df['CAR'], color='red', label='CAR acumulado', linewidth=2)
        ax2.set_ylabel('CAR acumulado')

        ax1.legend(loc='upper left')
        ax1.set_xlabel('Fecha')

        fechas = df['Date'].dt.strftime('%Y-%m-%d')
        event_date_str = pd.to_datetime(self.event.event_date).strftime('%Y-%m-%d')
        ax1.set_xticks(df['Date'])
        ax1.set_xticklabels(fechas, rotation=45, ha='right', fontsize=9)

        for label, fecha in zip(ax1.get_xticklabels(), fechas):
            label.set_color('red' if fecha == event_date_str else 'black')
            label.set_fontweight('bold' if fecha == event_date_str else 'normal')

        plt.title('Retornos Anormales (AR) y CAR acumulado')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_periodos_evento2(self, save_path=None):
        df = self.event._load_returns(self.event.ticker)
        t_event = pd.to_datetime(self.event.event_date)
        t_est_start = pd.to_datetime(self.event.estimation_start)
        t_est_end = pd.to_datetime(self.event.estimation_end)
        t_ev_start = pd.to_datetime(self.event.event_start)
        t_ev_end = pd.to_datetime(self.event.event_end)

        duracion_estimacion = (t_est_end - t_est_start).days + 1
        duracion_vacio = (t_ev_start - t_est_end).days - 1
        duracion_evento = (t_ev_end - t_ev_start).days + 1

        df = df[(df['Date'] >= t_est_start - pd.Timedelta(days=5)) & (df['Date'] <= t_ev_end + pd.Timedelta(days=5))]

        plt.figure(figsize=(14, 6))
        plt.plot(df['Date'], df['Return'], color='black', alpha=0.5, label='Retorno diario')

        plt.axvspan(t_est_start, t_est_end, color='skyblue', alpha=0.3, label='Estimación')
        plt.axvspan(t_est_end, t_ev_start, color='gray', alpha=0.1, label='Desacople')
        plt.axvspan(t_ev_start, t_ev_end, color='salmon', alpha=0.3, label='Evento')
        plt.axvline(t_event, color='red', linestyle='--', linewidth=1.2, label='Día del evento')

        y_pos = df['Return'].max() * 1.1

        def draw_bracket(x1, x2, text, color):
            xm = x1 + (x2 - x1) / 2
            plt.hlines(y=y_pos, xmin=x1, xmax=x2, color=color, linewidth=2)
            plt.text(xm, y_pos + 0.002, text, color=color, fontsize=10, ha='center')

        draw_bracket(t_est_start, t_est_end, f'Estimación ({duracion_estimacion} días)', 'blue')
        draw_bracket(t_est_end + pd.Timedelta(days=1), t_ev_start - pd.Timedelta(days=1), f'Desacople ({duracion_vacio} días)', 'gray')
        draw_bracket(t_ev_start, t_ev_end, f'Evento ({duracion_evento} días)', 'darkred')

        plt.title(f'Visualización de periodos clave del evento ({self.event.ticker})')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno logarítmico diario')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()