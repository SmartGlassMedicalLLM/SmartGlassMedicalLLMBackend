from inference.medgemma_utils import getLlm, convo_params

context_history = []

def prompt_convo(prompt):
    global context_history
    context_history.append(f"<start_of_turn>user\n{prompt}<end_of_turn>\n")
    full_prompt = "".join(context_history) + "<start_of_turn>model\n"

    outputs = getLlm().generate([full_prompt], convo_params)
    response = outputs[0].outputs[0].text
    
    context_history.append(f"<start_of_turn>model\n{response}<end_of_turn>\n")
    return response

def reset_convo():
    global context_history
    context_history = []