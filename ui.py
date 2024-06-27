import pandas as pd
from pandasai import SmartDataframe
import sqlite3
import chainlit as cl
from pandasai.llm.local_llm import LocalLLM

llm = LocalLLM(
    api_base = "http://localhost:11434/v1",
    model = "llama3:latest"
)

@cl.on_chat_start
def start_chat():
    #set initial message history
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assitant."}],
    )

@cl.on_message
async def main(message: cl.Message):
    #Retrieve message history
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    # Load data

    # SQL database connection
    # conn = sqlite3.connect('data.db')
    # df = pd.read_sql('SELECT * FROM shop', conn)
    # conn.close()

    # CSV 
    df=pd.read_csv('/Users/lagarto/Documents/AnalyticaSolutions/Portfolio/Data_Science/Pandas_AI/Coffee Shop Sales.csv')

    sdf= SmartDataframe(df, config={"llm": llm}, name='cofee_sales', description='This dataframe contains coffee sales transactions from three different locations')
    question = message.content
    response = sdf.chat(question)
    msg = cl.Message(content=response)

    await msg.send()

    # Update message history and send final message
    message_history.append({"role": "assistant", "content": msg})
    await msg.update()
