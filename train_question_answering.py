import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Adafactor
from transformers.optimization import AdafactorSchedule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

squad = load_dataset("squad", split="train[:5000]")

squad = squad.train_test_split(test_size=0.2)
train_ds = squad["train"]
test_ds = squad["test"]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


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

    torch.tensor(inputs, device=device)
    return inputs, inputs.input_ids


model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")


train_dataloader = DataLoader(train_ds, batch_size=64, collate_fn=preprocess_function)
test_dataloader = DataLoader(test_ds, batch_size=64, collate_fn=preprocess_function)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):

    tot_loss = 0
    for inputs, ids in train_dataloader:

        outputs = model(**inputs)
        pred_ids = outputs[:, outputs.start_logits.argmax():outputs.end_logits.argmax()+1]
        loss = loss_fn(pred_ids, ids)
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("loss: {:.3f}".format(tot_loss))