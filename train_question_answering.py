import collections

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

    return inputs


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


small_eval_set = test_ds.select(range(100))

eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=test_ds.column_names,
)

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

batches = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}

example_to_features = collections.defaultdict(list)

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)

data_collator = DefaultDataCollator()

train_dataloader = DataLoader(tokenized_squad["train"], batch_size=64, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_squad["test"], batch_size=1, collate_fn=data_collator)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(1):

    tot_loss = 0
    for inputs in train_dataloader:
        inputs = {key: value.to(device) for key, value in inputs.items()}
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


with torch.no_grad():
    outputs = model(**batches)

start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)

import numpy as np

n_best = 20
max_answer_length = 30
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

import evaluate

metric = evaluate.load("squad")

theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]

result = metric.compute(predictions=predicted_answers, references=theoretical_answers)
print("exact_match: {:.3f}, F-1: {:.3f}".format(result["exact_match"], result["f1"]))