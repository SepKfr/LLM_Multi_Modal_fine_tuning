import collections
import random
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from torch import nn

from models.fine_tune_question_answer import QuestionAnswerFineTune
from models.question_answer import QuestionAnswer
from process_data.Question_answer import QuestionAnswerData
from transformers import AutoModelForQuestionAnswering, Adafactor
from transformers.optimization import AdafactorSchedule

torch.random.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

squad = load_dataset("squad", split="train")

squad = squad.train_test_split(test_size=0.2)
train_ds = squad["train"]
test_ds = squad["test"]

qa_data = QuestionAnswerData(train=train_ds, test=test_ds)

model = QuestionAnswerFineTune().to(device)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

loss_fn = nn.CrossEntropyLoss()
best_model = None
train_best_loss = 1e10

for epoch in range(50):

    tot_loss = 0
    for inputs in qa_data.get_train_loader():
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(inputs)
        loss_start = loss_fn(outputs.start_logits, inputs["start_positions"])
        loss_end = loss_fn(outputs.end_logits, inputs["end_positions"])
        loss = loss_start + loss_end
        tot_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("epoch {}: loss: {:.3f}".format(epoch, tot_loss))
    if tot_loss < train_best_loss:
        train_best_loss = tot_loss
        best_model = model


batch, small_eval_set, eval_set = qa_data.get_test_data()

with torch.no_grad():
    outputs = model(batch)

start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

example_to_features = collections.defaultdict(list)

for idx, feature in enumerate(qa_data.get_eval_set()):
    example_to_features[feature["example_id"]].append(idx)

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
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

metric = evaluate.load("squad")

theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]

result = metric.compute(predictions=predicted_answers, references=theoretical_answers)
print("exact_match: {:.3f}, F-1: {:.3f}".format(result["exact_match"], result["f1"]))