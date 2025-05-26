import requests
import concurrent.futures

URL = "https://a643-2a02-a313-29f-6080-8fb-9d55-bbe0-5022.ngrok-free.app/chat"

payload = {
    "history": [
        {"role": "user", "message": "Czym jest Maszyna W?"},
        {"role": "assistant", "message": "Maszyna W to prosta, edukacyjna architektura komputera wykorzystywana do nauki programowania w asemblerze."},
        {"role": "user", "message": "Do czego służy instrukcja 'POB' w Maszynie W?"},
        {"role": "assistant", "message": "Instrukcja 'POB' ładuje wartość ze wskazanego adresu pamięci do akumulatora."},
        {"role": "user", "message": "Jak zsumować wszystkie elementy tablicy w asemblerze Maszyny W?"}
    ]
}

def send_request(i):
    try:
        response = requests.post(URL, json=payload, timeout=10000)
        if response.json().get('response'):
            return f"[{i}] ✅ Success  {response.json().get('response')[:50]}..."
        elif response.json().get('error'):
            return f"[{i}] ❌ Error {response.json().get('error')}"
    except Exception as e:
        return f"[{i}] ❌ Exception: {e}"

# How many parallel requests to fire
NUM_REQUESTS = 50

with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    futures = [executor.submit(send_request, i+1) for i in range(NUM_REQUESTS)]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
