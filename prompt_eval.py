import json
import pandas as pd
import requests
from textblob import TextBlob
import os
from dotenv import load_dotenv

# Load your Hugging Face token from .env file
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Load prompts
with open("prompts.json", "r") as f:
    prompts = json.load(f)

# Input query
query = input("Enter a chatbot query: ")

# Use Hugging Face text generation model (FLAN-T5 or Bloom)
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

results = []

for idx, template in enumerate(prompts):
    filled_prompt = template.format(query=query)

    # Send request to HF Inference API
    payload = {"inputs": filled_prompt}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        reply = result[0]["generated_text"].strip() if isinstance(result, list) else "Error: Invalid response"
    except Exception as e:
        reply = f"Error: {e}"

    sentiment = TextBlob(reply).sentiment.polarity
    length_penalty = abs(len(reply) - 200) / 200
    final_score = sentiment - length_penalty

    results.append({
        "Prompt #": idx + 1,
        "Prompt": filled_prompt,
        "Response": reply,
        "Sentiment Score": sentiment,
        "Final Score": final_score
    })

# Print results
for r in results:
    print(f"\nPrompt #{r['Prompt #']}:\n{r['Prompt']}")
    print(f"Response:\n{r['Response']}")
    print(f"Sentiment: {r['Sentiment Score']:.2f}, Score: {r['Final Score']:.2f}")
    print("-" * 50)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("huggingface_responses.csv", index=False)
print("\n✅ All results saved to huggingface_responses.csv")
