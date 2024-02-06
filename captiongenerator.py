from PIL import Image
from datasets import load_dataset
import itertools
import torch
from transformers import ViltProcessor, Trainer, pipeline
from transformers import DefaultDataCollator
from transformers import ViltForQuestionAnswering

dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])

labels = [item['ids'] for item in dataset['label']]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

def replace_ids(inputs):
  inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
  return inputs


dataset = dataset.map(replace_ids)
flat_dataset = dataset.flatten()

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")


def preprocess_data(examples):
    image_paths = examples['image_id']
    images = [Image.open(image_path) for image_path in image_paths]
    texts = examples['question']

    encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

    for k, v in encoding.items():
          encoding[k] = v.squeeze()
    targets = []

    for labels, scores in zip(examples['label.ids'], examples['label.weights']):
        target = torch.zeros(len(id2label))

        for label, score in zip(labels, scores):
            target[label] = score
        targets.append(target)

    encoding["labels"] = targets
    return encoding


processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])

data_collator = DefaultDataCollator()

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", num_labels=len(id2label), id2label=id2label, label2id=label2id)

from transformers import TrainingArguments

repo_id = "MariaK/vilt_finetuned_200"

training_args = TrainingArguments(
    output_dir=repo_id,
    per_device_train_batch_size=4,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset,
    tokenizer=processor,
)

trainer.train()

pipe = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_200")

example = dataset[0]
image = Image.open(example['image_id'])
question = example['question']
print(question)
pipe(image, question, top_k=1)

processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

image = Image.open(example['image_id'])
question = example['question']

# prepare inputs
inputs = processor(image, question, return_tensors="pt")

model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

example = dataset[0]
image = Image.open(example['image_id'])
question = example['question']

prompt = f"Question: {question} Answer:"

inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=10)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
