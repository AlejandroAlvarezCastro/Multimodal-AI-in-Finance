import pandas as pd
from event import Event

# Carga del DataFrame de conferencias (ejemplo)
df_conferences = pd.read_csv('conference_data.csv', index_col=0)    

event_summaries = []

for _, row in df_conferences.iterrows():
    try:
        event = Event(
            event_date=row['timestamp'],
            ticker=row['symbol'],
            year=str(row['year']),
            quarter=row['quarter'],
            plot_base_path='plots_eventos'
        )
        event.generate_and_save_plots()
        event_summaries.append(event.summary_dict())
    except Exception as e:
        print(f"❌ Error con {row['symbol']} en {row['timestamp']}: {e}")

df_summary = pd.DataFrame(event_summaries)
df_summary.to_csv("event_summaries.csv", index=False)
print("✅ Gráficos y resumen guardados correctamente.")
