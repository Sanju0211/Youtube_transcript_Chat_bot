# Youtube_transcript_Chat_bot

📺 LLM Video Helper
A Streamlit application that helps you summarize YouTube videos and answer questions about their content using local LLM (Ollama) and embeddings.


✨ Features
🎥 Extract transcripts from YouTube videos
📝 Generate concise summaries of video content
❓ Answer questions about the video content
🔍 Semantic search using embeddings
🏠 Runs locally with Ollama (no external API costs)
📥 Download summaries as text files

⚙️ Prerequisites
Before you begin, ensure you have the following installed:
Python 3.8 or higher
Ollama installed and running locally
Mistral model downloaded (ollama pull mistral)

🚀 Installation

Clone the repository:
git clone https://github.com/yourusername/llm-video-helper.git
cd llm-video-helper

Create and activate a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the dependencies:
pip install -r requirements.txt

Create a .env file in the root directory with the following content:
env
# Configuration for LLM Video Helper
# Note: This application currently uses local Ollama and doesn't require API keys
# Future API keys or sensitive configurations can be added here

# Example (not currently used):
# YOUTUBE_API_KEY=your_youtube_api_key_here

🏃‍♂️ Running the Application
First, ensure Ollama is running in another terminal:

ollama serve
In a separate terminal, run the Streamlit application:

streamlit run app.py
The application will open in your default browser at http://localhost:8501

🛠️ Usage
Enter a YouTube URL in the sidebar

The application will:
Fetch the video metadata (title, thumbnail)
Extract the transcript
Generate a summary

You can then:
Read the summary
Download it as a text file
Ask questions about the video content

🔧 Troubleshooting
Problem: Transcript cannot be fetched
✅ Solution: Not all YouTube videos have transcripts available. Try a different video.

Problem: Ollama connection error
✅ Solution: Ensure Ollama is running (ollama serve) and the Mistral model is downloaded (ollama pull mistral).

Problem: FAISS installation issues
✅ Solution: Try installing CPU version explicitly: pip install faiss-cpu

📂 Project Structure
text
llm-video-helper/
├── app.py                # Main application code
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── README.md             # This file
└── .gitignore           # Git ignore file
🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.



Made with ❤️ by Sanjeev 
