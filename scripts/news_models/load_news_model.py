import pandas as pd
import torch
import os
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#TODO: Make file path and save path as params to def 
def get_sentiment_score():
    #print(f'RUN FROM: {os.getcwd()}')  # Check where the script is being run from
    model_path = r"models/news_uncased_model"
    tokenizer_path = r"models/news_uncased_tokenizer"
    label_mapping = {'neutral': 0, 'negative': 1, 'positive': 2}
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = pd.read_csv("cryptopanic_web_scraper/output.csv") 
    titles = df["news"].tolist()
    batch_size = 32 #safe number
    predictions = []

    #Tokenize the dataset
    inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt")

    #Moving tensors to the correct device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    print("Starting predictions...")
    start_time = time.time()
    # Get predictions using batching, should be more efficient
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i+batch_size]
             #Tokenize the dataset
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            #Moving tensors to the correct device
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Get predicted class indices
            batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
            predictions.extend(batch_preds) 

    end_time = time.time()
    print("Predictions done!")
    print(f"Time take: {end_time - start_time:2f} seconds.")

    # Add predictions to DataFrame
    df["prediction"] = [reverse_mapping[pred] for pred in predictions]
    df.to_csv('scripts/news_models/predictions_2024.12.csv', index=False)
    print("\nPrediction saved to file")

get_sentiment_score()