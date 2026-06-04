from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage


load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)


messages = [SystemMessage(content="You are a helpful assistant. Answer the question as best as you can.")]

print("----------------------- welcome type 0 to exit the application-----------------------")
while True:
    prompt = input("You : ")
    
    if prompt == "0":
        break
        
   
    messages.append(HumanMessage(content=prompt))
    
    
    response = model.invoke(messages)
    print(f"AI: {response.content}")
    
    messages.append(AIMessage(content=response.content))
