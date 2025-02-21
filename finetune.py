from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import load_dataset

from torch import float32



# ✅ Charger le dataset JSONL
dataset = load_dataset('json', data_files='data/dataset.jsonl')

# ✅ Vérifier les clés du dataset
print("Exemple de données :", dataset['train'][0])  # Affiche un exemple pour vérifier sa structure

# ✅ Tokenizer et modèle
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA n'a pas de token de padding

# ✅ Fonction de prétraitement
def format_conversation(example):
    return {
        "text": "\n".join(f"{'Utilisateur' if msg['role'] == 'user' else 'Quera'}: {msg['content']}" 
                           for msg in example["messages"]).strip()
    }


# ✅ Appliquer la transformation
dataset = dataset.map(format_conversation)

# ✅ Fonction de prétraitement avec labels
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=40)
    
    # ✅ Copier input_ids pour labels
    inputs["labels"] = inputs["input_ids"].copy()

    # ✅ Remplacer les tokens de padding par -100 (Hugging Face ignore ces positions dans la perte)
    padding_token_id = tokenizer.pad_token_id
    inputs["labels"] = [
        [(label if label != padding_token_id else -100) for label in labels] for labels in inputs["labels"]
    ]

    return inputs



# ✅ Appliquer la transformation
dataset = dataset.map(preprocess_function, batched=True)


# ✅ Charger le modèle
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=float32, device_map="cpu")

# ✅ Config LoRA
lora_config = LoraConfig(
    r=32, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)


print("---------------------------------------------------------------")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("---------------------------------------------------------------")

training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    per_device_train_batch_size=5,  # ⚠️ Réduire la batch size car CPU est limité
    num_train_epochs=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir='./logs',
    learning_rate=2e-3,
    gradient_accumulation_steps=90,  # 🔹 Augmenter pour réduire la charge mémoire
    fp16=False,  # 🚫 Désactiver fp16 (inutile sur CPU)
    bf16=False,
    gradient_checkpointing=False,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    save_total_limit=2,
    weight_decay=0.01,
    report_to="tensorboard",
    torch_compile=False,  # ✅ Optimisation CPU
    no_cuda=True
)



# ✅ Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")


# ✅ Entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['train'],
    data_collator=data_collator
)

model.train(), trainer.train()

# ✅ Sauvegarde du modèle
model.save_pretrained("llama3-finetuned")
tokenizer.save_pretrained("llama3-finetuned")

print("🚀 Fine-tuning terminé avec succès ! 🎯")


# Charger le modèle de base et fusionner les poids LoRA
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=float32, device_map="cpu")
peft_model = PeftModel.from_pretrained(base_model, "llama3-finetuned")

# Fusionner les poids LoRA avec le modèle principal
peft_model = peft_model.merge_and_unload()

# Sauvegarde du modèle fusionné (sans LoRA)
peft_model.save_pretrained("llama3-finetuned-merged")
tokenizer.save_pretrained("llama3-finetuned-merged")

print("✅ Fusion et sauvegarde du modèle terminé !")
