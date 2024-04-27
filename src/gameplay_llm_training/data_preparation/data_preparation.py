def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### {example['instruction'][i]}\n ### Data: {example['data'][i]} ### Response: {example['label'][i]}"
        output_texts.append(text)
    return output_texts