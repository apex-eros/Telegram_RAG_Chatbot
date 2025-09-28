import evaluate
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    return full_text

def evaluate_rouge(predictions, references, questions):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)

    print("\nPer-question ROUGE scores:")
    for i, (q, pred, ref) in enumerate(zip(questions, predictions, references)):
        indiv = rouge.compute(predictions=[pred], references=[ref])
        print(f"Q{i+1}: {q}")
        print(f"  Model:     {pred}")
        print(f"  Gold:      {ref}")
        print(f"  ROUGE-1:   {indiv['rouge1']:.3f}")
        print(f"  ROUGE-2:   {indiv['rouge2']:.3f}")
        print(f"  ROUGE-L:   {indiv['rougeL']:.3f}")
        print(f"ROUGE-L F1: {results['rougeL']:.3f}")


    print("\nAggregate ROUGE:")
    print(f"ROUGE-1 F1: {results['rouge1']:.3f}")
    print(f"ROUGE-2 F1: {results['rouge2']:.3f}")
    print(f"ROUGE-L F1: {results['rougeL']:.3f}")

if __name__ == "__main__":
    pdf_path = "C:/Users/Admin/Desktop/JAYESH/Projects/Tg_chatbot/DS Interview_Preparation.pdf"  
    # Change this to your PDF file path

    print("Loading PDF...")
    text = load_pdf_text(pdf_path)
    print(f"The document has {len(text.split())} words.\n")

    # Fill your test QA pairs below
    questions = [
        "What is selection Bias?",
        "Who are False Positives?",
        "What are recommender systems?"
    ]

    # You decide the true answers and model answers (after running your RAG system)
    references = [
        "Selection bias occurs when the sample obtained is not representative of the population intended to be analysed.",
        "False Positives are the cases where you wrongly classified a non-event as an event a.k.a Type I error",
        "Recommender systems are a subclass of information filtering systems that are meant to predict the preferences or ratings that a user would give to a product"
    ]
    model_answers = [
        "Selection bias is a kind of error that occurs when the researcher decides who is going to be studied, usually associated with research where the selection of participants isn’t random. It is the distortion of statistical analysis, resulting from the method of collecting samples.",
        " Based on the context provided, False Positives are the cases where you wrongly classified a non-event as an event, also known as a Type I error",
        "Recommender systems are a subclass of information filtering systems that are meant to predict the preferences or ratings that a user would give to a product."
    ]

    evaluate_rouge(model_answers, references, questions)

