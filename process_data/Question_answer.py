import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QuestionAnswerData:

    def __init__(self, train, test, val=None, check_point="distilbert-base-uncased"):

        self.data_collator = DefaultDataCollator()
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)
        tokenized_train = train.map(self.preprocess_function, batched=True, remove_columns=train.column_names)
        self._train_dataloader = DataLoader(tokenized_train, batch_size=64, collate_fn=self.data_collator)

        if val is not None:
            tokenized_val = val.map(self.preprocess_function, batched=True, remove_columns=val.column_names)
            self._val_dataloader = DataLoader(tokenized_val, batch_size=64, collate_fn=self.data_collator)

        self._small_eval_set = test.select(range(100))
        trained_checkpoint = "distilbert-base-cased-distilled-squad"

        tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
        self._eval_set = self._small_eval_set.map(
            self.preprocess_validation_examples,
            batched=True,
            remove_columns=test["validation"].column_names,
        )

        self._eval_set_for_model = self._eval_set.remove_columns(["example_id", "offset_mapping"])
        self._eval_set_for_model.set_format("torch")

        self._batch = {k: self._eval_set_for_model[k].to(device) for k in self._eval_set_for_model.column_names}

    def get_train_loader(self):
        return self._train_dataloader

    def get_test_data(self):
        return self._batch, self._small_eval_set, self._eval_set

    def get_eval_set(self):
        return self._eval_set

    def get_val_loader(self):
        return self._val_dataloader

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
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

    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
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