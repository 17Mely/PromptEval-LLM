import json
import pandas as pd
import requests
from textblob import TextBlob
import os
from dotenv import load_dotenv
import streamlit as st

# Load your Hugging Face token from .env file
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Load prompts from JSON file
with open("prompts.json", "r") as f:
    prompts = json.load(f)

# Define the function to generate the best response
def generate_best_response(query):
    results = []
    
    for idx, template in enumerate(prompts):
        filled_prompt = template.format(query=query)
        payload = {"inputs": filled_prompt}

        try:
            response = requests.post("https://api-inference.huggingface.co/models/google/flan-t5-base", headers=headers, json=payload)
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

    # Find the best prompt based on the highest score
    best_response = max(results, key=lambda x: x["Final Score"])
    return best_response["Response"], best_response["Prompt"], best_response["Final Score"]

# Streamlit UI for user input
def streamlit_ui():
    st.title("PromptEval: LLM Prompt Tester")

    # Query input
    query = st.text_input("Enter your query:")

    if query:
        best_response, best_prompt, score = generate_best_response(query)
        st.write(f"**Best response for your query**:")
        st.write(f"**Prompt**: {best_prompt}")
        st.write(f"**Response**: {best_response}")
        st.write(f"**Sentiment Score**: {score:.2f}")
        st.write(f"**Final Score**: {score:.2f}")

        # Optionally save results to CSV
        if st.button('Save results to CSV'):
            df = pd.DataFrame([{
                "Prompt": best_prompt,
                "Response": best_response,
                "Sentiment Score": score,
                "Final Score": score
            }])
            df.to_csv('streamlit_responses.csv', index=False)
            st.success("Results saved to streamlit_responses.csv")

if __name__ == "__main__":
    streamlit_ui()
