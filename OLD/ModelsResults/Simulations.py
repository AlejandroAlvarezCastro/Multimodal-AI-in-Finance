import pandas as pd

from ModelsResults.QA.QASimulator import QASimulator
from ModelsResults.Monologues.MonologueSimulator import MonologueSimulator
from ModelsResults.SEC10K.SEC10KSimulator import TenKSimulator
from ModelsResults.Answered.AnsweredSimulator import AnsweredSimulator


results_df = pd.DataFrame(columns=["model_name", "elapsed_time", "accuracy", "precision", "recall", "f1_score"])

LLM_MODELS = [
        'deepseek-r1:1.5b', 'deepseek-r1:7b', 'deepseek-r1:8b', 'deepseek-r1:14b', #'deepseek-r1:32b',
        'llama3:8b', 
        'llama3.1:8b', 
        'llama3.2:1b', 
        'llama3.2:3b',
        'phi3:3.8b', 'phi3:14b',
        'phi4',
        'qwen2:0.5b', 'qwen2:1.5b', 'qwen2:7b',
        'qwen2.5:0.5b', 'qwen2.5:1.5b', 'qwen2.5:3b', 'qwen2.5:7b', 'qwen2.5:14b',
        'gemma:2b', 'gemma:7b',
        'gemma2:2b', 'gemma2:9b',
        'mistral'
    ]

def main():

    qa_labeled_data = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/TEXT/label_data/labeled_gold_sample_QA.csv"
    k10_labeled_data = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/clean_src/ModelsResults/SEC10K/labeled_gold_sample_10K.csv"
    monologues_labeled_data = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/clean_src/ModelsResults/Monologues/loabeled_gold_sample_monologues.csv"
    answered_labeled_data = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/2025_ACA_MultichannelAI/ModelsResults/Answered/qa_answering_labeled.json"

    qa_results_path = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/clean_src/ModelsResults/QA/results/"
    k10_results_path = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/clean_src/ModelsResults/SEC10K/results/"
    monologue_results_path = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/clean_src/ModelsResults/Monologues/results/"
    answered_results_path = "/home/aacastro/Alejandro/ACA_MultichanelAI_2025/2025_ACA_MultichannelAI/ModelsResults/Answered/results"

    print("Running QA simulation...")
    qa_sim = QASimulator(models=LLM_MODELS, labeled_data_path=qa_labeled_data, results_path=qa_results_path, results_df=results_df)
    qa_sim.run()

    print("Running 10-K simulation...")
    k10_sim = TenKSimulator(models=LLM_MODELS, labeled_data_path=k10_labeled_data, results_path=k10_results_path, results_df=results_df)
    k10_sim.run()

    print("Running Monologue simulation...")
    monologue_sim = MonologueSimulator(models=LLM_MODELS, labeled_data_path=monologues_labeled_data, results_path=monologue_results_path, results_df=results_df)
    monologue_sim.run()

    print("Running Answered simulation...")
    monologue_sim = AnsweredSimulator(models=LLM_MODELS, labeled_data_path=answered_labeled_data, results_path=answered_results_path, results_df=results_df)
    monologue_sim.run()

    print("All simulations completed.")

if __name__ == "__main__":
    main()

# Results (See ModelResults/): 

# Models for QA: ['llama3:8b', 'phi4', 'gemma2:9b']
# Models for Monologues: ['llama3:8b', 'llama3.1:8b', 'deepseek-r1:14b']
# Models for 10K: ['qwen2:7b', 'gemma2:9b', 'qwen2.5:7b']
# Models for Answered: ['']