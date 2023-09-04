import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from google.cloud import aiplatform

import os

# Configure a variável de ambiente
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\rita1\\Downloads\\chat-tupi-o-amigo-f652191e35cf.json"

vertexai.init(project="chat-tupi-o-amigo", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison@001")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
chat = chat_model.start_chat()

while True:
    # Solicita ao usuário que faça uma pergunta
    pergunta_do_usuario = input("user: ")

    if pergunta_do_usuario.lower() == 'sair':
        break  # Encerra o loop se o usuário digitar 'sair'

    # Envia a pergunta do usuário para o modelo
    response = chat.send_message(pergunta_do_usuario)

    # Imprime a resposta do modelo
    print(f"Resposta do Modelo: {response.text}")