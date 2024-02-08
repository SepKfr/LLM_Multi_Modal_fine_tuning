import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Adafactor
from transformers.optimization import AdafactorSchedule
from evaluate import load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

squad = load_dataset("squad", split="train[:5000]")

squad = squad.train_test_split(test_size=0.2)
train_ds = squad["train"]
test_ds = squad["test"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased").to(device)


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)

data_collator = DefaultDataCollator()

train_dataloader = DataLoader(tokenized_squad["train"], batch_size=64, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_squad["test"], batch_size=64, collate_fn=data_collator)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(1):

    tot_loss = 0
    for inputs in train_dataloader:
        inputs["input_ids"].to(device)
        inputs["attention_mask"].to(device)
        inputs["start_positions"].to(device)
        inputs["end_positions"].to(device)
        outputs = model(**inputs)
        loss_start = loss_fn(outputs.start_logits, inputs["start_positions"])
        loss_end = loss_fn(outputs.end_logits, inputs["end_positions"])
        loss = loss_start + loss_end
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("loss: {:.3f}".format(tot_loss))

wer = load("wer")
for inputs in test_dataloader:

    model.eval()
    inputs["input_ids"].to(device)
    inputs["attention_mask"].to(device)
    inputs["start_positions"].to(device)
    inputs["end_positions"].to(device)
    outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax(-1)
    answer_end_index = outputs.end_logits.argmax(-1)
    predict_answer_tokens = inputs.input_ids[answer_start_index: answer_end_index + 1]
    actual_answer_tokens = inputs.input_ids[inputs["start_positions"]:inputs["end_positions"]+1]
    predicted = tokenizer.decode(predict_answer_tokens)
    actual = tokenizer.decode(actual_answer_tokens)
    wer_score = wer.compute(predictions=predicted, references=actual)
    print("wer_score {:.3f}".format(wer_score))