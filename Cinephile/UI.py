import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Define the data structure
class Movie(BaseModel):
    title: str 
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

# Initialize LangChain components
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

model = load_llm()
parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract movie information from the paragraph.
{format_instructions}
"""),
    ("human", "{paragraph}")
])

# --- Streamlit UI Config ---
st.set_page_config(page_title="AI Movie Extractor", page_icon="🎬", layout="centered")

st.title("🎬 AI Movie Information Extractor")
st.write("Paste a paragraph, plot summary, or review below to extract structured movie data.")

# Text input area
para = st.text_area(
    "Enter Movie Paragraph:", 
    height=200, 
    placeholder="..."
)

# Extraction button
if st.button("Extract Data", type="primary"):
    if not para.strip():
        st.warning("Please enter some text before clicking extract.")
    else:
        with st.spinner("Analyzing text with Gemini..."):
            try:
                # Prepare prompt
                final_prompt = prompt.invoke({
                    "paragraph": para,
                    'format_instructions': parser.get_format_instructions()
                })

                # Call Model
                response = model.invoke(final_prompt)
                
                # Parse Result
                movie_data = parser.parse(response.content)

                # Display Results in organized components
                st.success("Extraction Successful!")
                
                st.header(f"🍿 {movie_data.title}")
                
                # Create grid columns for metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Release Year", value=movie_data.release_year or "N/A")
                with col2:
                    st.metric(label="Rating", value=f"⭐ {movie_data.rating}/10" if movie_data.rating else "N/A")
                with col3:
                    st.metric(label="Director", value=movie_data.director or "Unknown")

                # Genres & Cast tags
                st.write("**🎭 Genres:** " + ", ".join(movie_data.genre))
                st.write("**👥 Cast:** " + ", ".join(movie_data.cast))
                
                # Summary block
                st.info(f"**📝 Summary:**\n{movie_data.summary}")

                # Optional: Show Raw JSON output
                with st.expander("View Raw JSON Output"):
                    st.json(movie_data.model_dump())

            except Exception as e:
                st.error(f"An error occurred during parsing: {str(e)}")
                st.info("Tip: If the schema parsing failed, try tweaking the text inputs or lower the model temperature.")
