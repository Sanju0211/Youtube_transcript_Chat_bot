import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile

# --- Page setup ---
st.set_page_config(page_title="LLM Video Helper", page_icon="🎥", layout="wide")
st.title("🎥 LLM Video Helper")

# --- Layout columns ---
left_col, right_col = st.columns([2, 3])

# --- Input: YouTube URL ---
with st.sidebar:
    st.markdown("### 📺 Enter YouTube video URL")
    video_url = st.text_input("YouTube URL", "https://youtu.be/iBF8x8rvnsU")

# --- Model setup ---
llm = OllamaLLM(model="mistral")

# --- Prompt templates ---
summary_prompt = PromptTemplate.from_template(
    "Summarize the following video transcript in a concise paragraph:\n\n{context}"
)

qa_prompt = PromptTemplate.from_template(
    "Given the following transcript, answer the question.\n\nTranscript:\n{context}\n\nQuestion: {question}"
)

# --- Helper: Extract video ID ---
def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
    except Exception:
        return None

# --- Helper: Fetch transcript ---
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        return f"Error fetching transcript: {e}"

# --- Helper: Fetch metadata ---
def fetch_video_metadata(video_id):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.replace(" - YouTube", "").strip()
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
        return title, thumbnail_url
    except Exception as e:
        return "Title not found", ""

# --- Main logic ---
if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
    else:
        title, thumbnail_url = fetch_video_metadata(video_id)

        with left_col:
            st.subheader(f"📺 {title}")
            if thumbnail_url:
                st.image(thumbnail_url, use_column_width=True)

        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)

        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            st.success("Transcript fetched successfully!")

            # Split into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents([transcript])

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            with tempfile.TemporaryDirectory() as tmpdirname:
                db = FAISS.from_documents(docs, embeddings)
                retriever = db.as_retriever()

                # --- Video Summary ---
                with right_col:
                    with st.spinner("Summarizing the video..."):
                        summary_chain = create_stuff_documents_chain(llm=llm, prompt=summary_prompt)
                        summary = summary_chain.invoke({"context": docs})
                        st.markdown("### 📝 Video Summary")
                        st.info(summary)

                        # Downloadable summary
                        st.download_button(
                            label="📥 Download Summary as TXT",
                            data=summary,
                            file_name=f"{title}_summary.txt",
                            mime="text/plain"
                        )

                # --- Q&A Section ---
                st.markdown("### ❓ Ask a Question about the Video")
                query = st.text_input("Your question:")
                if query:
                    with st.spinner("Searching for answer..."):
                        relevant_docs = retriever.invoke(query)
                        qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
                        response = qa_chain.invoke({
                            "context": relevant_docs,
                            "question": query
                        })
                        st.markdown("**Answer:**")
                        st.success(response)
