import torch
from medgemma_utils import tokenizer, model

context_history = []

def prompt_convo(prompt):
    global context_history
    
    context_history.append(f"<start_of_turn>user\n{prompt}<end_of_turn>\n")
    
    full_prompt = "".join(context_history) + "<start_of_turn>model\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.01,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    context_history.append(f"<start_of_turn>model\n{response}<end_of_turn>\n")
    
    return response

def reset_convo():
    global context_history
    context_history = []