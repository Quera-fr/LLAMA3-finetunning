from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import load_dataset
import torch

# ✅ Charger le dataset JSONL
dataset = load_dataset('json', data_files='data/dataset.jsonl')

# ✅ Vérifier les clés du dataset
print("Clés du dataset :", dataset['train'].column_names)
print("Exemple de données :", dataset['train'][0])  # Vérification de la structure

# ✅ Tokenizer et modèle
model_name = "ministral/Ministral-3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Ajustement du token de padding

# ✅ Fonction de prétraitement avec labels
def format_conversation(messages):
    formatted_text = ""  # Initialisation du texte formaté
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        formatted_text += f"{role}: {content}\n"
    return formatted_text

def preprocess_function(examples):
    formatted_inputs = [format_conversation(msg) for msg in examples["messages"]]
    inputs = tokenizer(formatted_inputs, truncation=True, padding="max_length", max_length=50)
    
    # Copier input_ids pour labels
    inputs["labels"] = inputs["input_ids"].copy()

    # Remplacer les tokens de padding par -100
    padding_token_id = tokenizer.pad_token_id
    inputs["labels"] = [
        [(label if label != padding_token_id else -100) for label in labels] for labels in inputs["labels"]
    ]
    return inputs

# ✅ Appliquer la transformation
dataset = dataset.map(preprocess_function, batched=True)

# ✅ Charger le modèle
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)

# ✅ Config LoRA (Mistral-specific layers)
lora_config = LoraConfig(
    r=32, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

print("---------------------------------------------------------------")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("---------------------------------------------------------------")

training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=2,  # ⚠️ Ajusté pour Mistral (plus lourd que LLaMA-1B)
    num_train_epochs=25, 
    logging_dir='./logs',
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
model.save_pretrained("mistral-finetuned")
tokenizer.save_pretrained("mistral-finetuned")

print("🚀 Fine-tuning terminé avec succès ! 🎯")

# Charger le modèle de base et fusionner les poids LoRA
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
peft_model = PeftModel.from_pretrained(base_model, "mistral-finetuned")

# Fusionner les poids LoRA avec le modèle principal
peft_model = peft_model.merge_and_unload()

# Sauvegarde du modèle fusionné (sans LoRA)
peft_model.save_pretrained("mistral-finetuned-merged")
tokenizer.save_pretrained("mistral-finetuned-merged")

print("✅ Fusion et sauvegarde du modèle terminé !")
