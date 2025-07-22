import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file at the very beginning
load_dotenv()

# --- Hugging Face API Token Check and LLM Setup ---
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_api_token:
    st.error("Hugging Face API token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable in your .env file.")
    st.stop()

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,
        max_new_tokens=500,
        huggingfacehub_api_token=hf_api_token,
    )
)

# --- Page setup ---
st.set_page_config(page_title="LLM Video Helper", page_icon="🎥", layout="wide")
st.title("🎥 LLM Video Helper")

# --- Layout columns ---
left_col, right_col = st.columns([2, 3])

# --- Input: YouTube URL ---
with st.sidebar:
    st.markdown("### 📺 Enter YouTube video URL")
    video_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# --- Prompt templates ---
summary_prompt = PromptTemplate.from_template(
    "You are a helpful assistant that summarizes video transcripts. Summarize the following video transcript in a concise paragraph:\n\n{context}"
)

qa_prompt = PromptTemplate.from_template(
    "You are a helpful assistant that answers questions based on video transcripts. Given the following transcript, answer the question.\n\nTranscript:\n{context}\n\nQuestion: {question}"
)

# --- Helper: Extract video ID ---
def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        # Standard YouTube watch URL: https://www.youtube.com/watch?v=VIDEO_ID, www.youtube.com/watch?v=, etc.
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'www.youtube.com', 'm.youtube.com']:
            query_v = parse_qs(parsed_url.query).get('v')
            if query_v:
                return query_v[0]
        # Shortened URL: https://youtu.be/VIDEO_ID
        elif parsed_url.hostname in ['youtu.be']:
            return parsed_url.path[1:]
        return None
    except Exception:
        return None

# --- Helper: Fetch transcript ---
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        return f"Error fetching transcript: {e}. Please ensure the video has English captions and is publicly accessible."

# --- Helper: Fetch metadata ---
def fetch_video_metadata(video_id):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}" # Use actual YouTube domain here
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find("meta", property="og:title")
        title = title_tag["content"] if title_tag else "Video Title Not Found"

        thumbnail_tag = soup.find("meta", property="og:image")
        thumbnail_url = thumbnail_tag["content"] if thumbnail_tag else ""

        return title, thumbnail_url
    except requests.exceptions.RequestException as e:
        st.warning(f"Network or HTTP error fetching video metadata. It might be a private/deleted video or an invalid ID. Error: {e}")
        return "Title not found", ""
    except Exception as e:
        st.warning(f"Could not parse video metadata. Error: {e}")
        return "Title not found", ""

# --- Main logic ---
if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL. Please enter a valid YouTube video link (e.g., https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID).")
    else:
        title, thumbnail_url = fetch_video_metadata(video_id)

        with left_col:
            st.subheader(f"📺 {title}")
            if thumbnail_url:
                st.image(thumbnail_url, use_container_width=True)

        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)

        if transcript.startswith("Error"):
            st.error(transcript)
        elif not transcript.strip():
            st.warning("Transcript fetched successfully, but it appears to be empty. This might happen for videos without captions.")
        else:
            st.success("Transcript fetched successfully!")

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents([transcript])

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            with tempfile.TemporaryDirectory() as tmpdirname:
                db = FAISS.from_documents(docs, embeddings)
                retriever = db.as_retriever()

                # --- Video Summary ---
                with right_col:
                    with st.spinner("Summarizing the video..."):
                        try:
                            summary_chain = create_stuff_documents_chain(llm=llm, prompt=summary_prompt)
                            summary = summary_chain.invoke({"context": docs})
                            st.markdown("### 📝 Video Summary")
                            st.info(summary)

                            st.download_button(
                                label="📥 Download Summary as TXT",
                                data=summary,
                                file_name=f"{title}_summary.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Error generating summary: {e}. This might be a temporary API issue or prompt content problem.")
                            summary = "Could not generate summary due to an error."
                            st.info(summary) # Display a fallback message


                # --- Q&A Section ---
                st.markdown("### ❓ Ask a Question about the Video")
                query = st.text_input("Your question:")
                if query:
                    with st.spinner("Searching for answer..."):
                        try:
                            relevant_docs = retriever.invoke(query)
                            qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
                            response = qa_chain.invoke({
                                "context": relevant_docs,
                                "question": query
                            })
                            st.markdown("**Answer:**")
                            st.success(response)
                        except Exception as e:
                            st.error(f"Error generating answer: {e}. This might be a temporary API issue or prompt content problem.")
                            st.markdown("**Answer:** Could not generate answer due to an error.")
