# 🌍 Bond AI Travel Planner

Bond is an AI-powered travel planning assistant built with **LangGraph**, **Streamlit**, and a multi-agent architecture.

The application helps users plan trips by combining flights, hotels, activities, transportation, visa information, weather, and travel recommendations into a single interactive experience.

---

## ✨ Features

* 🤖 Multi-Agent Architecture (Planner, Executor, Replanner, Critic)
* 👤 Human-in-the-Loop (HITL) approvals and decision points
* 🎚️ Multiple autonomy levels (SIF)
* 💾 Persistent conversations using LangGraph checkpoints and SQLite
* 🧳 Structured travel cards (Flights, Hotels, Activities, Transport, Budget)
* 🔄 Multiple conversations per user
* 🌙 Light / Dark mode
* 📊 Live planning progress tracking
* 🌐 Web search integration for travel information
* 💬 Modern chat-style user interface

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <repository_url>
cd Agentic-Travel-Planner
```

### 2. Create a virtual environment

```bash
python -m venv taenv
```

Activate it:

**Windows**

```bash
taenv\Scripts\activate
```

**Linux / macOS**

```bash
source taenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_key
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## 🛠️ Technologies

* Python
* Streamlit
* LangGraph
* LangChain
* SQLite
* Google Gemini
* Groq

---

## 🎓 Academic Project

This project demonstrates the use of:

* Agentic AI systems
* Multi-agent collaboration
* Planning and execution workflows
* Human-in-the-Loop interaction
* Conversational user interfaces
* Persistent memory and state management

---

## 📌 Notes

* Internet connection may be required for external travel information.
* API keys are not included in the repository.
* Chat history is stored locally.
* The application is intended for educational and demonstration purposes.
