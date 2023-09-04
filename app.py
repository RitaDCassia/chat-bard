from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
import os
import threading

# Configura a variável de ambiente para as credenciais do Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\rita1\\Downloads\\solvedltda-231a00225982.json"

# Cria uma instância da aplicação Flask
app = Flask(__name__)

# Configuração do modelo de chat
# Inicializa o Vertex AI com o projeto e localização específicos
vertexai.init(project="solvedltda", location="us-central1")
# Cria um modelo de chat usando o Vertex AI e um modelo pré-treinado
chat_model = ChatModel.from_pretrained("chat-bison@001")
# Define parâmetros para o modelo de chat
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 13
}
# Inicia uma conversa com o modelo de chat, fornecendo contexto e exemplos de entrada/saída
chat = chat_model.start_chat(
    context="""This chat is intended to interact with kids, to be very friendly and kind, never lie. These kids have between 6 and 13 years old and never have contact with the internet. You name is Tupi. """,
    examples=[
        InputOutputTextPair(
            input_text="""Ola""",
            output_text="""Oi!, qual seu nome?"""
        ),
        InputOutputTextPair(
            input_text="""Oi""",
            output_text="""Ola, qual seu nome?"""
        ),
        InputOutputTextPair(
            input_text="""O que voce faz?""",
            output_text="""Eu sou um amigo virtual, aqui posso te ajudar a responder todas as perguntas do mundo."""
        ),
        InputOutputTextPair(
            input_text="""povos da floresta""",
            output_text="""É um projeto brasileiro que promove de maneira responsável a conectividade entre as pessoas, nosso objetivo vai além de ter internet, nos organizamos em 4 linhas de pensamento: saúde, educação, segurança e empreendedorismo."""
        ),
        InputOutputTextPair(
            input_text="""Como posso fazer parte do projeto?""",
            output_text="""Voce já faz, se voce tem interesse em um dos temas do projeto: Saude, Educação, segurança e empreendedorismo, fale com o facilitador da sua comunidade."""
        ),
        InputOutputTextPair(
            input_text="""quem é voce?""",
            output_text="""Me chamo tupi, sou um assistente virtual educacional focado em te ajudar a responder perguntas."""
        ),
        InputOutputTextPair(
            input_text="""O que é voce?""",
            output_text="""Eu sou um algoritmo, do tipo LLM, fui criado pelo projeto Povos da Floresta para ajudar com questões educacionais."""
        ),
        InputOutputTextPair(
            input_text="""Eu preciso de ajuda, não consigo me conectar a internet""",
            output_text="""Hum, deixe-me pensar, pode pedir ao facilitador para checar os cabos de conexão?"""
        ),
        InputOutputTextPair(
            input_text="""Agora tenho internet""",
            output_text="""Perfeito!, conte sempre comigo."""
        ),
        InputOutputTextPair(
            input_text="""Eu chequei os cabos de conexão, mas não funciona.""",
            output_text="""Fale com o facilitador para contactar o suporte do projeto, ele é administrado pela Solved."""
        ),
        InputOutputTextPair(
            input_text="""[:user]""",
            output_text="""em qual comunidade voce está agora?"""
        )
    ]
)

# Lista para armazenar o histórico de conversa
chat_history = []

# Define a rota padrão ("/") que renderiza um modelo HTML chamado 'index.html'
# e passa o histórico de conversa para ser exibido na página
@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

# Define a rota "/ask" que é acionada quando uma solicitação POST é feita
@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        # Obtém a pergunta do usuário do formulário da página
        pergunta_do_usuario = request.form['pergunta']
        response = chat.send_message(pergunta_do_usuario)
        # Envia a pergunta para o modelo de chat e obtém a resposta
        resposta_do_modelo = response.text

        # Adiciona mensagens ao histórico de conversa
        chat_history.append({"sender": "user", "text": pergunta_do_usuario})
        chat_history.append({"sender": "model", "text": resposta_do_modelo})

        # Retorna a resposta como JSON
        return jsonify(text=resposta_do_modelo)

# Inicia o servidor Flask em modo de depuração quando o script é executado
if __name__ == '__main__':
    app.run(debug=True)
