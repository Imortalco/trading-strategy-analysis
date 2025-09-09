import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, DatasetDict

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print('Loading data ...')
#print(os.getcwd())
data = pd.read_csv('data/crypto_news/cryptonews_dataset.csv')

# Map sentiment labels to numerical values
print('Mapping labels...')
label_mapping = {'Neutral': 0, 'Negative': 1, 'Positive': 2}
data['label'] = data['sentiment'].map(label_mapping)

print('Splitting data ...')
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'].tolist(), 
    data['label'].tolist(),
    test_size = 0.2,
    random_state=42
)

train_data = {'text': train_texts, 'label': train_labels}
test_data = {'text': test_texts, 'label': test_labels}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)
datasets = DatasetDict({"train": train_dataset, "test": test_dataset})


import gc 

def train_model(model_name):
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3, ignore_mismatched_sizes=True)
        model.to(device)

        tokenized_datasets = datasets.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir = f'./results_{model_name}',
            num_train_epochs= 3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            warmup_ratio=0.01,
            logging_dir=f'./logs_{model_name}',
            logging_steps=10,
            eval_strategy='epoch'
        )

        trainer = Trainer(model=model,
                        args=training_args,
                        train_dataset=tokenized_datasets['train'],
                        eval_dataset=tokenized_datasets['test'],
                        data_collator=data_collator)
        
        trainer.train()
        print(f'After training {model_name}, CUDA memory summary:')
        print(torch.cuda.memory_summary())
    
        torch.cuda.empty_cache()  # Clear GPU memory after training
        del tokenizer, tokenized_datasets #Deleting unused variables 
        gc.collect()  # Collect garbage
        return model, trainer
    
    except Exception as e:
        # Handle exceptions
        print(f"An error occurred while training model {model_name}: {e}")
        return None, None
    
#models = ['ElKulako/cryptobert', 
#'kk08/CryptoBERT', 
#'ProsusAI/finbert', 
#'FacebookAI/roberta-base', 
#'microsoft/deberta-v3-base', 
#'google-bert/bert-base-cased',
#'google-bert/bert-base-uncased',
#'distilbert/distilbert-base-cased']
models = ['google-bert/bert-base-uncased']
# import multiprocessing as mp

# mp.set_start_method('spawn', force=True)

trained_models = {}

for model_name in models:
    print(f'Training model: {model_name}')
    model, trainer = train_model(model_name)

    if model is not None:
        trained_models[model_name] = (model, trainer)
    else:
        print(f"Skipping model {model_name} due to training error.")
    
    # Clean up after each model to free memory
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

def save_mistakes_to_txt(mistakes, filename):
     # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

     # Open the file and write the mistakes
    with open(filename, "w") as file:
        for index, text, true_label, predicted_label, confidence in mistakes:
            file.write(f"Index: {index}, Text: {text}, True Label: {true_label}, Predicted Label: {predicted_label}, Confidence: {confidence:.4f}\n")
    print(f"Mistakes saved to {filename}")

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, tokenizer, test_texts, test_labels, model_name):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Tokenize the test dataset
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

    # Create a test DataLoader
    test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    model.eval()
    model.to(device)
    predictions = []
    confidences = []
    mistakes = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            confidences.extend(probs.cpu().numpy()) #Store the whole probability dristribution 

    #Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions) 
    
    #Identify mistakes and their confidence
    for i in range(len(test_labels)):
        if test_labels[i] != predictions[i]:
            predicted_confidence = confidences[i][predictions[i]]  # Get confidence for the predicted class
            mistakes.append((i, test_texts[i], test_labels[i], predictions[i], predicted_confidence))


    if mistakes:
        file_path = f"mistakes/mistakes_{model_name}.txt"
        save_mistakes_to_txt(mistakes, file_path)
        print(f"Mistakes made by the model : {len(mistakes)}")
        #for index, text, true_label, predicted_label, confidence in mistakes:
            #print(f"Index: {index}, Text: {text}, True Label: {true_label}, Predicted Label: {predicted_label}, Confidence: {confidence:.4f}")

     # Optionally, you can also print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, digits=4))

    return accuracy, predictions, confidences, mistakes

def compare_model_mistakes(model_mistakes, model_predictions, test_texts, test_labels):
    if len(model_mistakes) < 2:
        print("Not enough models to compare mistakes.")
        return
    
    model_names = list(model_mistakes.keys())
    mistakes_first_model = {mistake[0] for mistake in model_mistakes[model_names[0]]}  # Indices of mistakes for the first model
    mistakes_second_model = {mistake[0] for mistake in model_mistakes[model_names[1]]}  # Indices of mistakes for the first model
    
    common_mistakes = mistakes_first_model.intersection(mistakes_second_model)
    unique_mistakes_first_model = mistakes_first_model - common_mistakes
    unique_mistakes_second_model = mistakes_second_model - common_mistakes
    
    if common_mistakes:
        print(f"\nCommon Mistakes: {len(common_mistakes)}")
        for idx in common_mistakes:
            print(f"Index: {idx}, Text: {test_texts[idx]}, True Label: {test_labels[idx]}, {model_names[0]} Predicted: {model_predictions[model_names[0]][idx]}, {model_names[1]} Predicted: {model_predictions[model_names[1]][idx]}")

    if unique_mistakes_first_model:
        print(f"\nUnique Mistakes of {model_names[0]}: {len(unique_mistakes_first_model)}")
        for idx in unique_mistakes_first_model:
            mistake = [m for m in model_mistakes[model_names[0]] if m[0] == idx][0]
            print(f"Index: {mistake[0]}, Text: {mistake[1]}, True Label: {mistake[2]}, Predicted: {mistake[3]}, Confidence: {mistake[4]:.4f}")
    
    if unique_mistakes_second_model:
        print(f"\nUnique Mistakes of {model_names[1]}: {len(unique_mistakes_second_model)}")
        for idx in unique_mistakes_second_model:
            mistake = [m for m in model_mistakes[model_names[1]] if m[0] == idx][0]
            print(f"Index: {mistake[0]}, Text: {mistake[1]}, True Label: {mistake[2]}, Predicted: {mistake[3]}, Confidence: {mistake[4]:.4f}")

print(f"Evaluating models start...")
# Evaluate each trained model

model_predictions = {}
model_confidences = {}
model_mistakes = {}

for model_name, (model, trainer) in trained_models.items():
    print(f'\nEvaluating {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    accuracy, predictions, confidences, mistakes = evaluate_model(model, tokenizer, test_texts, test_labels, model_name)
    model_predictions[model_name] = predictions
    model_confidences[model_name] = confidences
    model_mistakes[model_name] = mistakes
    print(f"Accuracy of {model_name}: {accuracy:.4f}")
    
# Compare mistakes between the models (min 2)
# compare_model_mistakes(model_mistakes, model_predictions, test_texts, test_labels)
