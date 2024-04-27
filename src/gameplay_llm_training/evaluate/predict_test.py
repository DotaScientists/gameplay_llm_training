from transformers import LlamaForCausalLM, LlamaTokenizerFast
from datasets import load_from_disk
from gameplay_llm_training.settings import  Settings
import torch
from torch.utils.data import DataLoader
from gameplay_llm_training.data_preparation.data_preparation import  formatting_prompts_func



def get_predictions():
    settings = Settings()
    model = LlamaForCausalLM.from_pretrained(settings.local_training_output_path, device_map="cuda")

    tokenizer = LlamaTokenizerFast.from_pretrained(
        settings.pretrained_model_name,
        cache_dir=settings.local_cache_path
    )
    test_dataset = load_from_disk(str(settings.local_test_dataset_path))

    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=1500,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    tokenized_dataset = test_dataset.map(
        tokenize,
        batched=True,
        num_proc=8,
        batch_size=1000,
    )
    device = torch.device("cuda")
    dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        input_ids_batch = batch["input_ids"]
        attention_mask_batch = batch["attention_mask"]

        input_ids = torch.tensor(input_ids_batch).to(device)
        attention_mask = torch.tensor(attention_mask_batch).to(device)
        print("Input ids", input_ids.shape)

        outputs = model.generate(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            max_length=1500,
            num_return_sequences=1,
            do_sample=True,
            temperature=1,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            num_beams=5,
            early_stopping=True,
            use_cache=True,
        )
        print("Output", outputs)
        decoded_text = tokenizer.decode(outputs.detach()[0], skip_special_tokens=True)
        print("Decoded text", decoded_text)
        break


if __name__ == "__main__":
    get_predictions()