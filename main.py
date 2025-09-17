import os
import csv
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import streamlit as st
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

#load environment variable
load_dotenv()

#Initialize Parser
parser = StrOutputParser()

#Initialize Model 
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

#Schema for validating sentiment output
class Sentiment(BaseModel):
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(
        description='The sentiment of the transcript'
        )

#UI Code
st.header("Divyam's Assignment")
UserInput= st.text_area("Input Transcript", height=200)

#Prompts
summary_prompt = PromptTemplate(
    template=(
        "Summarize the following customer call transcript in 2–3 sentences. "
        "Keep it concise and factual.\n\nTranscript:\n{transcript}"
    ),
    input_variables=["transcript"]
)

sentiment_prompt = PromptTemplate(
    template=
             "You are a strict sentiment classifier for customer service calls. \n"
        "Base your answer ONLY on the customer's experience, not politeness. \n"
        "If the customer reports a problem, failure, frustration, or urgency → classify as Negative. \n"
        "If the customer is fully satisfied and happy → classify as Positive. \n"
        "If neither applies clearly → classify as Neutral. \n\n"
        "Return ONLY JSON in this exact format: {{\"sentiment\": \"Positive\"}}\n\n"
        "Transcript:\n{transcript}",
    input_variables=['transcript']
    )


if st.button('Get Sentiment'):
    if not UserInput:
        st.warning('Please provide the transcript')
        st.stop()

    with st.spinner('Fetching and Analyzing...'):
        try:
            #Summarization code
            parallelChain = RunnableParallel({
                'Summary': summary_prompt | model | parser,
                'Sentiment': sentiment_prompt | model | parser

            })
           
            result = parallelChain.invoke({'transcript': UserInput})
            
            summary_result = (result['Summary']).strip()
            raw_sentiment = (result['Sentiment']).strip()

            try:
                validate = Sentiment.model_validate_json(raw_sentiment)
                sentiment_result = validate.sentiment
            except ValidationError:
                sentiment_result= "Unknown"
                st.error("Model did not return valid sentiment")
                st.write("Raw result from model:", raw_sentiment)
                     

            #Display result
            st.subheader("Summary")
            st.info(summary_result)

            st.subheader("Sentiment")
            st.success(sentiment_result)
            
           
            #Save to csv
            file_path = "call_analysis.csv"
            file_exist = os.path.isfile(file_path)

            with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if not file_exist:
                    writer.writerow(["Transcript", "Summary", "Sentiment"])
                writer.writerow([UserInput, summary_result, sentiment_result])

            st.caption(f"saved to {file_path}")
        
        except Exception as e:
            st.error(f'An error occurred: {e}.')    