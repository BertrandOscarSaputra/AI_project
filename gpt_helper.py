import os
from openai import OpenAI

# Pastikan Anda sudah set OPENAI_API_KEY di environment variable
client = OpenAI()

def get_explanation(prediction):
    prompt = f"Jelaskan dalam bahasa Indonesia apa arti diagnosis '{prediction}' pada kanker kulit."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # atau model lain jika Anda memiliki akses
        messages=[
            {"role": "system", "content": "Kamu adalah pakar medis yang menjelaskan penyakit."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()
