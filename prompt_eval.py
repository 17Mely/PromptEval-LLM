from openai import OpenAI
import json
import csv
from textblob import TextBlob
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
client = OpenAI()  # Automatically picks up OPENAI_API_KEY from .env

# Load prompt templates from prompts.json
with open("prompts.json", "r") as f:
    prompt_templates = json.load(f)

# Take a chatbot query from the user
query = input("Enter a chatbot query: ")

# Function to score the response based on sentiment and length
def score_response(response):
    sentiment = TextBlob(response).sentiment.polarity
    length_penalty = abs(len(response) - 200) / 200  # Ideal ~200 chars
    final_score = sentiment - length_penalty
    return sentiment, final_score

# Store results
results = []

# Loop through each prompt
for idx, template in enumerate(prompt_templates):
    prompt = template.format(query=query)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Error: {e}"

    sentiment, final_score = score_response(reply)
    results.append((idx + 1, prompt, reply, sentiment, final_score))

# Print results to terminal
print("\n--- Prompt Evaluation Results ---\n")
for res in results:
    print(f"Prompt #{res[0]}:\n{res[1]}\n\nResponse:\n{res[2]}\nSentiment: {res[3]:.2f}, Score: {res[4]:.2f}")
    print("-" * 50)

# Save results to responses.csv
with open("responses.csv", "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Prompt #", "Prompt", "Response", "Sentiment Score", "Final Score"])
    for res in results:
        writer.writerow(res)

print("\n✅ All results saved to responses.csv")
