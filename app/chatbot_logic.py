import torch

def generate_response(model, tokenizer, conversation_history, user_input):
    conversation_history.append(f"User: {user_input}\nAssistant:")

    conversation = "\n".join(conversation_history)

    inputs = tokenizer(conversation, return_tensors="pt", padding=True, truncation=True).to("cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    latest_response = response.split("Assistant:")[-1].strip()

    conversation_history.append(f"Assistant: {latest_response}\n")

    return latest_response, conversation_history
