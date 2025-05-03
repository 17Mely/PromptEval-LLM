import openai
import json
from textblob import TextBlob

# Set your OpenAI API key
openai.api_key = "your-api-key-here"

# Load prompt templates
with open("prompts.json", "r") as f:
    prompt_templates = json.load(f)

# Define query to test
query = input("Enter a chatbot query: ")

# Store results
results = []

# Send each prompt to LLM
for idx, template in enumerate(prompt_templates):
    prompt = template.format(query=query)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    reply = response.choices[0].message['content']
    sentiment = TextBlob(reply).sentiment.polarity

    results.append((idx+1, prompt, reply.strip(), sentiment))

# Display results
print("\n--- Prompt Evaluation Results ---\n")
for res in results:
    print(f"Prompt #{res[0]}:\n{res[1]}\nResponse: {res[2]}\nSentiment Score: {res[3]:.2f}\n{'-'*40}")
