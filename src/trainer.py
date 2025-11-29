import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

df = pd.read_csv("frida_finetune.csv", sep="|").sample(frac=1, random_state=42)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

model_name = "ai-forever/FRIDA"
model = SentenceTransformer(model_name)

model.max_seq_length = 256

model[0].auto_model.gradient_checkpointing_enable()

# 4. Подготовка данных
train_examples = []
for idx, row in train_df.iterrows():
    train_examples.append(InputExample(texts=[str(row['search_query']), str(row['search_document'])]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
sentences1 = [str(row['search_query']) for i, row in test_df.iterrows()]
sentences2 = [str(row['search_document']) for i, row in test_df.iterrows()]
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, [1.0]*len(sentences1))

print(f"Начинаем обучение. Max seq length: {model.max_seq_length}")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    optimizer_params={"lr": 5e-5},
    warmup_steps=int(len(train_dataloader) * 0.1),
    evaluator=evaluator,
    evaluation_steps=50,
    output_path="./fine_tuned_frida_opt",
    save_best_model=True,
    use_amp=True,
    show_progress_bar=True
)
