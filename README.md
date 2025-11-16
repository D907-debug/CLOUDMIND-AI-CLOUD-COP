# 🌩️ CloudMind – AI Cloud Cop
### *Autonomous Cloud Governance • Risk Detection • Smart Actions*

CloudMind is an **AI-powered cloud automation and security system** designed to detect risks, score cloud activity, automate safe actions, monitor cloud health, and present everything through a simple console/dashboard.

It is built for:
- 🚀 Hackathons  
- 🧠 AI + Cloud enthusiasts  
- 🛡️ Cloud security monitoring  
- ⚙️ DevOps teams  


## 🚀 Features

### 🔍 1. AI-Based Risk Detection  
- Detects suspicious cloud patterns  
- Scores severity using ML/logic rules (`risk_scorer.py`)  
- Tracks all events in `actions_log.json`

### 🤖 2. Automated Decision Engine  
- Reads rules from `action_rules.json`  
- Decides what action to take (`decision_engine.py`)  
- Supports safe fallback actions

### ⚡ 3. Cloud Automation Engine  
- Executes cloud actions like resource cleanup, monitoring, preventive fixes  
- Controlled via `main.py`

### 📊 4. Dashboard & Cloud Console  
- `cloud_console.py` → Command-line cloud interface  
- `monitor_dashboard.py` → Live monitoring dashboard

### 🗃️ 5. Data-Driven Structure  
- Rule-based system using JSON  
- Logs each action  
- Uses `.env` for environment configs  


## 📂 Project Structure

```
cloudmind/
│
├── backend/
│   ├── decision_engine.py        # AI engine to choose actions
│   ├── risk_scorer.py            # Ranks severity of issues
│   ├── main.py                   # Main controller orchestrator
│   ├── teaching.py               # Rule updates / learning logic
│   │
│   └── data/
│       ├── action_rules.json     # Rules for automated actions
│       └── actions_log.json      # History/log of executed actions
│
├── frontend/
│   ├── cloud_console.py          # User CLI console
│   └── monitor_dashboard.py      # System monitoring UI
│
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables
```


## 🧪 How It Works

1. **Risk Detection**  
   Backend analyzes cloud inputs & assigns a risk score.

2. **Decision Engine**  
   Based on score + JSON rules, the system chooses the best action.

3. **Execution Layer**  
   Performs cleanup, alerts, scaling, etc.

4. **Monitoring**  
   Dashboards display logs and real-time state.


## ▶️ Setup & Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Add your environment variables
Create `.env`:

```
CLOUD_API_KEY=your_key
CLOUD_ENV=development
```

### 3️⃣ Run the engine
```bash
python backend/main.py
```

### 4️⃣ Launch cloud console
```bash
python frontend/cloud_console.py
```

### 5️⃣ Run monitoring dashboard
```bash
python frontend/monitor_dashboard.py
```


## ⚙️ Tech Stack
- Python 3.10+  
- JSON Rule Engine  
- CLI + Dashboard  
- Event Logging  


## 🛠️ Future Enhancements
- Web UI (Streamlit/React)  
- Cloud provider integration (AWS/GCP/Azure)  
- ML anomaly detection  
- Notification system  


## 👨‍💻 Author
CloudMind — Built with ❤️ for hackathons.
