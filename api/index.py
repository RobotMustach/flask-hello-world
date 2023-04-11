from tkinter import *
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import os

# Crea la ventana principal
root = Tk()
root.title("Chatbot")

# Crea el área de texto para la conversación
conversation_area = Text(root, height=20, width=50)
conversation_area.pack()

# Crea el área de entrada para el usuario
user_input = Entry(root, width=50)
user_input.pack()

# Construye el índice
def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

index = construct_index("context_data/data")

# Define la función para enviar la respuesta del chatbot
def send_message(event=None):
    # Obtiene la entrada del usuario
    user_text = user_input.get()
    conversation_area.insert(END, "You: " + user_text + "\n")

    # Consulta el índice para obtener la respuesta del chatbot
    response = index.query(user_text).response

    # Muestra la respuesta del chatbot en el área de conversación
    conversation_area.insert(END, "Chatbot: " + response + "\n")

    # Limpia la entrada del usuario
    user_input.delete(0, END)

# Crea el botón para enviar la entrada del usuario
send_button = Button(root, text="Send", command=send_message)
send_button.pack()

# Permite que la tecla Enter envíe la entrada del usuario
root.bind('<Return>', send_message)

# Ejecuta la aplicación
root.mainloop()
