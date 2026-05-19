from src.evaluation.retriever_evaluator import RetrieverEvaluator

def main():
    evaluator = RetrieverEvaluator(top_k=5)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
