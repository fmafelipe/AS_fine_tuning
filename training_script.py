from datasets import Dataset, Value, ClassLabel, Features
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers import Trainer, logging 
import pandas as pd
import numpy as np
import torch
import evaluate


# hiperparametros
lr = 7e-5
train_batch = 8
eval_batch = 8
num_epochs = 3


# Cargar dataset

mi_dataset_df =pd.read_csv("dataset.csv", sep=";")
mi_dataset_df = mi_dataset_df.drop(["estrellas","idrev_places","estrellas","fecha_q","fecha_rev","id_place","usuarios_iduser","num palabras"], axis=1)
features = Features({"text": Value("string"), "label": ClassLabel(num_classes=2, names=["negativo","positivo"])})
mi_dataset = Dataset.from_pandas(mi_dataset_df, features=features)
mi_dataset = mi_dataset.shuffle(seed=42)
mi_dataset = mi_dataset.train_test_split(test_size=0.2, shuffle= False)


# Tokenizar

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

def tokenizer_function(mi_dataset):
    return tokenizer(mi_dataset["text"], padding = "max_length", truncation=True, return_attention_mask= True, return_token_type_ids= False) 

tokenized_dataset = mi_dataset.map(tokenizer_function, batched=True)
small_train_dataset = tokenized_dataset["train"].select(range(80))
small_eval_dataset = tokenized_dataset["test"].select(range(20))



# importar modelo el cual se va a ajustar

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=2).to(device)


logging.set_verbosity_error()


training_args = TrainingArguments(
    output_dir="resultados_ajuste",
    evaluation_strategy="epoch",
    learning_rate = lr,
    per_device_train_batch_size = train_batch,
    per_device_eval_batch_size = eval_batch,
    num_train_epochs= num_epochs,
    save_strategy = "epoch"
    
)


#calcular metrica 

metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  

#instacionar trainer
trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset = small_train_dataset,
    eval_dataset = small_eval_dataset,
    compute_metrics = compute_metrics,
)



def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

#entrenar 
result = trainer.train()
print_summary(result)




print("termino")
