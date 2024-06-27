import chainlit as cl

from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn


llm = LocalLLM(
    api_base = "http://localhost:11434/v1",
    model = "llama3:latest"
)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hello").send()
    files = None
    # Waits for user to upload CSV file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload CSV file to begin", accept=["text/csv"], max_size_mb=100
        ).send()

    # Load the CSV data and store in user session
    file =files[0]

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file.path)
    
    # Creating user session to store data
    cl.user_session.set("message_history", 
                        [{"role": "system", "content": "You are a helpful assistant."}])
    cl.user_session.set("data", df)
    
    # Send response back to user
    await cl.Message(
        content= f"{file.name} uploaded! You can ask me anything related to your CSV"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    #Retrieve message history
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    
    # Get data
    df =cl.user_session.get("data")
    sdf= SmartDataframe(df, config={"llm": llm})
    question = message.content
    response = sdf.chat(question)
    msg = cl.Message(content=response)

    await msg.send()

    # Update message history and send final message
    message_history.append({"role": "assistant", "content": msg})
    cl.user_session.set("message_history", message_history)
    await msg.update()