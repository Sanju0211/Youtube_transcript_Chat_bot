# Youtube_transcript_Chat_bot
# 🎥 LLM Video Helper

A Streamlit application designed to help you interact with YouTube video content using Large Language Models (LLMs). This tool can extract video transcripts, summarize them, and answer your questions, offering flexibility with different LLM backends.

---

## ✨ Features

* **📺 YouTube Transcript Extraction:** Automatically fetches transcripts for specified YouTube video URLs.
* **📝 Video Summarization:** Generates a concise summary of the video's content using an LLM.
* **❓ Interactive Q&A:** Allows you to ask questions about the video content and receive LLM-powered answers.
* **🔍 Semantic Search:** Leverages embeddings (via `sentence-transformers` and FAISS) for efficient context retrieval during Q&A.
* **📥 Downloadable Summaries:** Provides an option to download the generated summary as a text file.
* **Flexible LLM Backends:**
    * **`app.py`**: Configured for **local LLM inference with Ollama** (ideal for privacy and avoiding API costs).
    * **`app_v1.py`**: Configured for **Hugging Face Hub models** (requires a Hugging Face API token for remote inference).

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8 or higher**
* **Git** (for cloning the repository)

### For Ollama Usage (`app.py`):

* **Ollama installed and running locally.** Download from [ollama.ai](www.youtube.com6).
* **A compatible model downloaded via Ollama CLI** (e.g., `mistral`).
    ```bash
    ollama pull mistral
    ```

### For Hugging Face Usage (`app_v1.py`):

* **Hugging Face API Token** with at least "read" access for model inference. You can generate one from your [Hugging Face Settings -> Access Tokens](www.youtube.com5).

---

## 🚀 Installation

Follow these steps to get the project up and running on your local machine:

### 1. Clone the Repository

Open your terminal or command prompt and clone the project:

```bash
git clone [https://github.com/your-username/LLM_Video_Helper.git](https://github.com/your-username/LLM_Video_Helper.git)
cd LLM_Video_Helper
````

*(Remember to replace `your-username/LLM_Video_Helper.git` with your actual repository URL if you've uploaded it to GitHub).*

\<br\>

### 2\. Create and Activate a Virtual Environment (Recommended)

It's best practice to use a virtual environment to manage project dependencies:

```bash
python -m venv myenv
```

**Activate the virtual environment:**

  * **Windows:**
    ```bash
    .\myenv\Scripts\activate
    ```
  * **macOS/Linux:**
    ```bash
    source myenv/bin/activate
    ```
    *(You'll see `(myenv)` preceding your command prompt, indicating the environment is active.)*

\<br\>

### 3\. Install Python Dependencies

Install all necessary libraries using pip. It's recommended to first generate a `requirements.txt` from a working environment if you don't have one, or install them manually:

```bash
# Option 1: If you have a requirements.txt (recommended after initial setup)
pip install -r requirements.txt

# Option 2: Manual Installation (if requirements.txt is not yet available)
# These are the core libraries needed for both app.py and app_v1.py
pip install streamlit requests beautifulsoup4 youtube-transcript-api langchain langchain-huggingface langchain-community python-dotenv sentence-transformers

# Install the CPU-only version of PyTorch for embeddings (crucial for performance/compatibility)
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

\<br\>

### 4\. Configure Your `.env` File

This file stores sensitive information like API tokens.

1.  In the root directory of your project (where `app.py` and `app_v1.py` are located), create a new file named **`.env`** (make sure it starts with a dot).

2.  Open the `.env` file and add the following content:

    ```env
    # Configuration for LLM Video Helper
    # This application uses a Hugging Face API token for app_v1.py
    # and local Ollama for app.py (no API key needed for Ollama itself).

    # For Hugging Face Hub models (used by app_v1.py)
    HUGGINGFACEHUB_API_TOKEN="hf_YOUR_ACTUAL_HUGGING_FACE_TOKEN_GOES_HERE"

    # Example: You can add other API keys here if needed in the future
    # OPENAI_API_KEY="sk-YOUR_OPENAI_KEY"
    ```

    **Replace `"hf_YOUR_ACTUAL_HUGGING_FACE_TOKEN_GOES_HERE"` with your actual Hugging Face Access Token.**

-----

## 🏃‍♂️ Running the Application

Choose the version you want to run based on your LLM backend preference:

### Option A: Run with Local Ollama (`app.py`)

This option processes LLM requests directly on your machine using Ollama.

1.  **Start Ollama Server:** Open a *separate* terminal or command prompt and start the Ollama server. Keep this window open.
    ```bash
    ollama serve
    ```
2.  **Run Streamlit App:** In your *activated virtual environment terminal*, run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

### Option B: Run with Hugging Face Hub (`app_v1.py`)

This option sends LLM requests to Hugging Face's inference endpoints.

1.  **Verify `.env`:** Ensure your `HUGGINGFACEHUB_API_TOKEN` is correctly set in your `.env` file.
2.  **Run Streamlit App:** In your *activated virtual environment terminal*, run the Streamlit application:
    ```bash
    streamlit run app_v1.py
    ```

After running either command, the application will open automatically in your default web browser at `http://localhost:8501`.

-----

## 🛠️ Usage

1.  **Enter a YouTube URL:** Paste the URL of the YouTube video you want to analyze into the input field in the sidebar.
2.  **Process:** The application will then:
      * Fetch the video's title and thumbnail.
      * Extract the complete transcript of the video.
      * Generate a concise summary of the content.
3.  **Interact:**
      * Read the generated summary directly on the page.
      * Download the summary as a `.txt` file using the provided button.
      * Ask specific questions about the video content in the "Ask a Question" section to get detailed answers.

-----

## 🔧 Troubleshooting

  * **Problem: `Hugging Face API token not found` (for `app_v1.py`)**

      * ✅ **Solution:** Ensure you have created a `.env` file in the project's root directory and `HUGGINGFACEHUB_API_TOKEN` is correctly defined within it. Double-check for typos.

  * **Problem: `Error fetching transcript` / Video has no captions.**

      * ✅ **Solution:** Not all YouTube videos have publicly available transcripts (especially older or non-English videos). Try a different video with known captions.

  * **Problem: Ollama connection error / Model not found (for `app.py`)**

      * ✅ **Solution:** Verify that the `ollama serve` command is running in a separate terminal. Also, confirm that the required model (e.g., `mistral`) has been downloaded via `ollama pull mistral`.

  * **Problem: `NotImplementedError: Cannot copy out of meta tensor` / Embedding model issues.**

      * ✅ **Solution:** This indicates a conflict with PyTorch and `sentence-transformers`. Try a clean reinstallation:
        ```bash
        pip uninstall torch torchaudio torchvision -y
        pip uninstall sentence-transformers -y
        pip cache purge
        pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
        pip install sentence-transformers
        ```

  * **Problem: `TypeError: 'NoneType' object is not iterable` / LLM API call failures (for `app_v1.py`)**

      * ✅ **Solution:** This usually means the Hugging Face API returned an unexpected or empty response.
          * Check your internet connection.
          * Verify your Hugging Face API token is valid and active.
          * The API might be experiencing temporary rate limits or server issues. Wait a few minutes and try again.

-----

## 📂 Project Structure

```
llm-video-helper/
├── app.py                  # Main application using Ollama (local LLM)
├── app_v1.py               # Main application using Hugging Face Hub LLMs
├── .env                    # Environment variables (e.g., Hugging Face API Token)
├── requirements.txt        # Python dependencies for both versions
├── README.md               # This file
└── .gitignore              # Specifies intentionally untracked files to ignore
```

-----

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to open an issue or submit a pull request.

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

-----

Made with ❤️ by Sanjeev

```
```
