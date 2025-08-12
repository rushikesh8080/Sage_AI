from huggingface_hub import login

your_token = "hf_token" 
login(token=your_token)

import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ✅ Load LLaMA Model
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ Load JSON File
INPUT_FILE = "filtered_second_entry.json"
OUTPUT_FILE = "dialogue_sample.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

# ✅ Function to Extract All Passages from JSON Properly
def extract_passages(data):
    """ Extracts section texts from nested JSON keys (URLs > Sections > Paragraphs) """
    passages = []
    
    for url, sections in data.items():  # Iterate through URLs
        for section_title, paragraphs in sections.items():  # Iterate through sections
            if isinstance(paragraphs, list):
                section_text = " ".join(paragraphs)  # Merge paragraph list into a single passage
                passages.append((section_title, section_text))
    
    return passages

# ✅ Function to Generate Structured Dialogues
def generate_dialogues(section_text):
    prompt = f'''
    You are an AI philosopher with deep knowledge of aesthetics and the paradox of tragedy.
    Generate structured, multi-turn dialogues based on the passage below.

    - Extract diverse User questions related to the passage.
    - Provide concise AI responses that flow naturally in a conversation.
    - Maintain a formal and insightful tone.
    - Ensure at least 10 Q&A pairs.

    ### Output Format ###
    User Question: <question>
    AI Response: <response>

    Passage:
    """{section_text}"""  

    ### START DIALOGUE ###
    '''

    response = generator(
        prompt,
        max_new_tokens=1024,
        num_return_sequences=1,
        temperature=0.3,
        repetition_penalty=1.3,
        do_sample=True,
        return_full_text=False  # Ensures only generated content is returned
    )

    generated_text = response[0]['generated_text'] if response else ""
    
    # Debug: Check what the model generated
    print("\n🔍 Raw Generated Output:\n", generated_text, "\n")

    return generated_text

# ✅ Function to Extract and Refine Q&A Pairs
def clean_generated_text(response_text):
    """ Extract structured Q&A pairs from generated text """
    qa_pairs = []

    # Ensure text is not empty
    if not response_text.strip():
        print("⚠️ Warning: Generated response is empty!")
        return qa_pairs
    
    # Extract Q&A pairs using regex
    matches = re.findall(r"User Question:\s*(.*?)\nAI Response:\s(.*?)(?=\nUser Question:|$)", response_text, re.DOTALL)

    if not matches:
        print("⚠️ Warning: No valid Q&A pairs found in generated text!")
    
    for match in matches:
        user_question = match[0].strip()
        ai_response = match[1].strip()
        if user_question and ai_response:
            qa_pairs.append({"role": "user", "content": user_question})
            qa_pairs.append({"role": "assistant", "content": ai_response})
    
    return qa_pairs

# ✅ Process All Sections
chat_data = []  # Store dialogues for all sections

for section_title, section_text in extract_passages(articles):
    response_text = generate_dialogues(section_text)  # Generate raw text
    qa_pairs = clean_generated_text(response_text)  # Extract structured Q&A pairs
    
    if qa_pairs:
        conversation = {
            "messages": [
                {"role": "system", "content": "You are a philosopher with knowledge from the Stanford Encyclopedia of Philosophy. Answer questions concisely and thoughtfully, maintaining a natural conversation flow."}
            ] + qa_pairs
        }
        chat_data.append(conversation)

# ✅ Save the Output to a JSON File
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chat_data, f, indent=4, ensure_ascii=False)

print(f"✅ Chat dataset saved to {OUTPUT_FILE}!")
