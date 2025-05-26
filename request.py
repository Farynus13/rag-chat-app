import requests

response = requests.post("https://real-large-cricket.ngrok-free.app/chat", json={
    "history": [
        {"role": "user", "message": "Czym jest Maszyna W?"},
        {"role": "assistant", "message": "Maszyna W to prosta, edukacyjna architektura komputera wykorzystywana do nauki programowania w asemblerze."},
        {"role": "user", "message": "Do czego służy instrukcja 'POB' w Maszynie W?"},
        {"role": "assistant", "message": "Instrukcja 'POB' ładuje wartość ze wskazanego adresu pamięci do akumulatora."},
        {"role": "user", "message": "Jak zsumować wszystkie elementy tablicy w asemblerze Maszyny W?"}
    ]
})

print(response.json()["response"])
