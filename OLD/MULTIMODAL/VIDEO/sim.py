from VideoEmotionAnalizer import VideoEmotionAnalyzer

import sys
import os
import time
import pandas as pd

# Agregar el directorio raíz del proyecto al path
sys.path.append(os.path.abspath(".."))

from ..ModelMetrics import ModelMetrics

# DATASET = 'AFEW'
DATASET = 'RAVDESS'

ROWS_PER_EMOTION = 50
skip_values = [0.1, 0.25, 0.5, 0.75, 1]
models = ['vit', 'fer'] # emotieff: ['enet_b0_8_best_afew.pt', 'enet_b0_8_best_vgaf.pt', 'enet_b0_8_va_mtl.pt', 'enet_b2_8_best.pt'],
methods = ["mode", "mean", "abs"]



results_df = pd.DataFrame(columns=["model_name", "elapsed_time", "accuracy", "precision", "recall", "f1_score"])
df = pd.read_csv(f'/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/VIDEO/skip_results/{DATASET}.csv', index_col=0)
df['label'] = df['label'].str.lower()
label_mapping = {"calm": "neutral", "fearful": "fear", "surprised": "surprise"}
df["label"] = df["label"].replace(label_mapping)
df_sampled = df.groupby("label").sample(n=ROWS_PER_EMOTION, random_state=35)
df = df_sampled


df_results = pd.DataFrame()

for model in models:
    model_results_df = pd.DataFrame()  # DataFrame específico por modelo
    for skip in skip_values:
        for method in methods:
            print(f"Probando skips={skip}, method={method} con modelo={model}")

            try:
                model_name = f"{model}_{skip}_{method}"
                results_path = f'/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/VIDEO/POO/sim_results/{DATASET}/{model}/'

                emotion_analysis = VideoEmotionAnalyzer(mode=model, skips=skip, method=method)
                evaluator = ModelMetrics(model_name=model_name, results_path=results_path, model=model)

                start_time = time.time()
                df = emotion_analysis.classify_dataframe(df)
                elapsed_time = round(time.time() - start_time, 2)
                
                # Obtener métricas
                metrics = evaluator.get_results(df)
                
                # Crear DataFrame con los resultados de esta iteración
                iteration_results_df = pd.DataFrame([
                    {"model_name": model_name, "model": model, "skip": skip, "method": method, "elapsed_time": elapsed_time, **metrics}
                ])
                
                # Agregar a los DataFrames correspondientes
                model_results_df = pd.concat([model_results_df, iteration_results_df], ignore_index=True)
                df_results = pd.concat([df_results, iteration_results_df], ignore_index=True)

            except Exception as e:
                print(f"An error occurred with model {model_name}: {e}")
                continue
    
    # Guardar CSV por modelo
    model_results_df.to_csv(results_path + f"/metrics_{model}_{DATASET}.csv", index=False)
    
    # Visualización para el modelo actual
    if not model_results_df.empty:
        evaluator.plot_f1_vs_time_all_models(model_results_df)
    
    print('-'*100)

# Guardar el DataFrame global al final
df_results.to_csv(f"/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/VIDEO/POO/sim_results/metrics_global_{DATASET}.csv", index=False)
evaluator.plot_f1_vs_time_all_models(df_results)