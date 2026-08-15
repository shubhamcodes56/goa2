import requests
import pyttsx3
import json

# Text-to-Speech (Voice) setup
engine = pyttsx3.init()
engine.setProperty('rate', 160) # Bolne ki speed
voices = engine.getProperty('voices')
# Optional: Try to set a female voice for Siri feel
for voice in voices:
    if "Zira" in voice.name or "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

def ask_siri():
    print("\n" + "="*50)
    print("🤖 AI Assistant: Main apki kya madad kar sakti hu?")
    
    # 1. Take User Input
    query = input("Aap (Type your question): ")
    if query.lower() in ['exit', 'quit', 'stop']:
        print("🤖 AI: Alvida! Have a great day.")
        engine.say("Alvida! Have a great day.")
        engine.runAndWait()
        exit()
        
    print("🤖 AI: Hmm, main apne 15 Lakh dimaag ke panno me dhundh rahi hu...")
    
    # 2. Ask the "Brain" (Our Local API)
    try:
        response = requests.post("http://localhost:8000/ask", json={"query": query})
        
        if response.status_code == 200:
            data = response.json()
            best_answer = data.get("answer", "")
            
            if best_answer:
                # Print karna
                print(f"\n🗣️ AI Answer: {best_answer}")
                print(f"⚡ [Latency: Total {data['latency_ms']:.0f}ms | Qdrant {data['retrieval_ms']:.0f}ms | Groq {data['llm_generation_ms']:.0f}ms]\n")
                
                # 3. Speak the Answer! (Text-to-Speech)
                engine.say(best_answer)
                engine.runAndWait()
            else:
                print("🤖 AI: Mujhe iska jawab database me nahi mila.")
                engine.say("Sorry, I could not find the answer.")
                engine.runAndWait()
        else:
            print("🤖 AI: Mera API se connection toot gaya hai!")
            
    except requests.exceptions.ConnectionError:
        print("🤖 AI: Error! Lagta hai aapka Uvicorn API server on nahi hai.")
        print("Pehle 'uvicorn benchmark_api:app --host 0.0.0.0 --port 8000' run karein.")

if __name__ == "__main__":
    print("Siri/Alexa Demo Start ho gaya hai! (Exit karne ke liye 'quit' type karein)")
    while True:
        ask_siri()
