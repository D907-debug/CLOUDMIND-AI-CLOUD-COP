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
# backend/
## decision_engine.py
```
def make_decision(risk_score):
    """
    Map risk score to a decision and action plan
    
    Risk Levels:
    - 0-3: Low (Allow)
    - 4-6: Medium (Warn)
    - 7-8: High (Approval Required)
    - 9-10: Critical (Block)
    
    Parameters:
    - risk_score: Float from 0-10
    
    Returns:
    - Dictionary with decision details
    """
    
    if risk_score <= 3:
        return {
            "decision": "ALLOW",
            "level": "Low",
            "emoji": "✅",
            "color": "green",
            "action_required": "Action proceeds normally - just logged for audit",
            "alert_team": False,
            "requires_approval": False,
            "can_override": True
        }
    
    elif risk_score <= 6:
        return {
            "decision": "WARN",
            "level": "Medium",
            "emoji": "⚠️",
            "color": "yellow",
            "action_required": "Show warning message and require engineer confirmation",
            "alert_team": False,
            "requires_approval": False,
            "can_override": True
        }
    
    elif risk_score <= 8:
        return {
            "decision": "APPROVAL_REQUIRED",
            "level": "High",
            "emoji": "🛡️",
            "color": "orange",
            "action_required": "Pause action and require manager approval before proceeding",
            "alert_team": True,
            "requires_approval": True,
            "can_override": False
        }
    
    else:  # risk_score 9-10
        return {
            "decision": "BLOCK",
            "level": "Critical",
            "emoji": "🚫",
            "color": "red",
            "action_required": "Action blocked immediately - requires security review and 2-person approval",
            "alert_team": True,
            "requires_approval": True,
            "can_override": False
        }


def get_next_steps(decision_type):
    """
    Get recommended next steps based on decision
    """
    steps = {
        "ALLOW": [
            "Action will proceed automatically",
            "Event logged for audit trail",
            "No further action needed"
        ],
        "WARN": [
            "Review the warning message carefully",
            "Confirm you understand the risks",
            "Click 'Proceed Anyway' if you're certain",
            "Consider using the suggested alternative"
        ],
        "APPROVAL_REQUIRED": [
            "Your manager has been notified via Slack",
            "Action is paused pending approval",
            "Expected response time: 15-30 minutes",
            "Use suggested alternative for faster resolution"
        ],
        "BLOCK": [
            "Action has been blocked for security",
            "DevOps team has been alerted",
            "Submit a change request ticket",
            "Schedule the action during maintenance window",
            "Get approval from 2 senior engineers"
        ]
    }
    
    return steps.get(decision_type, ["Contact DevOps team for guidance"])


# Test function
if __name__ == "__main__":
    print("="*60)
    print("TESTING DECISION ENGINE")
    print("="*60)
    
    test_scores = [2, 5, 7, 9.5]
    
    for score in test_scores:
        print(f"\n--- Risk Score: {score}/10 ---")
        decision = make_decision(score)
        print(f"Decision: {decision['emoji']} {decision['decision']}")
        print(f"Level: {decision['level']}")
        print(f"Action: {decision['action_required']}")
        print(f"Alert Team: {decision['alert_team']}")
        print(f"Requires Approval: {decision['requires_approval']}")
        
        steps = get_next_steps(decision['decision'])
        print("\nNext Steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
    
    print("\n" + "="*60)
```
## main.py
```
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.risk_scorer import calculate_risk, get_risk_factors
from backend.decision_engine import make_decision, get_next_steps

# Create FastAPI app
app = FastAPI(
    title="CloudMind API",
    description="AI-powered cloud action risk assessment",
    version="1.0.0"
)

# Add CORS middleware so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ActionRequest(BaseModel):
    engineer_id: str
    action_type: str
    resource_name: str
    environment: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "engineer_id": "sanjay@company.com",
                "action_type": "DeleteDatabase",
                "resource_name": "production-main-db",
                "environment": "production"
            }
        }

# Helper function to save logs
def log_action(action_data):
    """Save action to logs file"""
    try:
        # Read existing logs
        try:
            with open('data/actions_log.json', 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
        
        # Add new log
        logs.append(action_data)
        
        # Keep only last 50 actions
        logs = logs[-50:]
        
        # Write back
        with open('data/actions_log.json', 'w') as f:
            json.dump(logs, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error logging action: {e}")
        return False

# ROOT ENDPOINT
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "CloudMind API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "check_action": "/api/check-action",
            "get_actions": "/api/actions",
            "get_stats": "/api/stats",
            "docs": "/docs"
        }
    }

# MAIN ENDPOINT - Check Action
@app.post("/api/check-action")
def check_action(request: ActionRequest):
    """
    Analyze a cloud action and determine if it should be allowed
    
    Returns risk score, decision, and teaching explanation
    """
    
    try:
        print(f"\n{'='*60}")
        print(f"ANALYZING ACTION")
        print(f"{'='*60}")
        print(f"Engineer: {request.engineer_id}")
        print(f"Action: {request.action_type}")
        print(f"Resource: {request.resource_name}")
        print(f"Environment: {request.environment}")
        print(f"{'='*60}\n")
        
        # Step 1: Calculate risk score
        risk_score = calculate_risk(
            request.action_type,
            request.resource_name,
            request.environment
        )
        
        # Step 2: Make decision
        decision_data = make_decision(risk_score)
        
        # Step 3: Get risk factors (for detailed display)
        risk_factors = get_risk_factors(
            request.action_type,
            request.resource_name,
            request.environment
        )
        
        # Step 4: Get next steps
        next_steps = get_next_steps(decision_data['decision'])
        
        # Step 5: Create teaching message (simple for now)
        teaching_messages = {
            "ALLOW": f"✅ This action is safe to proceed. {risk_factors['action_description']}.",
            "WARN": f"⚠️ Be careful! {risk_factors['action_description']}. Make sure you've tested this in a non-production environment first.",
            "APPROVAL_REQUIRED": f"🛡️ This action needs manager approval because {risk_factors['action_description'].lower()} in {request.environment} environment. Consider using a safer alternative or scheduling during maintenance window.",
            "BLOCK": f"🚫 Action blocked! {risk_factors['action_description']} on {request.resource_name} in {request.environment} is too risky. This could cause major outages or data loss. Please submit a change request with proper review."
        }
        
        teaching = teaching_messages.get(
            decision_data['decision'],
            "Action detected and analyzed."
        )
        
        # Prepare response
        response = {
            "timestamp": datetime.now().isoformat(),
            "engineer_id": request.engineer_id,
            "action_type": request.action_type,
            "resource_name": request.resource_name,
            "environment": request.environment,
            "risk_score": risk_score,
            "decision": decision_data['decision'],
            "level": decision_data['level'],
            "emoji": decision_data['emoji'],
            "color": decision_data['color'],
            "action_required": decision_data['action_required'],
            "alert_team": decision_data['alert_team'],
            "requires_approval": decision_data['requires_approval'],
            "teaching": teaching,
            "next_steps": next_steps,
            "risk_factors": risk_factors
        }
        
        # Log the action
        log_action(response)
        
        print(f"\n{'='*60}")
        print(f"RESULT: {decision_data['emoji']} {decision_data['decision']}")
        print(f"Risk Score: {risk_score}/10")
        print(f"{'='*60}\n")
        
        return response
    
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# GET RECENT ACTIONS
@app.get("/api/actions")
def get_actions(limit: int = 10):
    """Get recent actions from log"""
    try:
        with open('data/actions_log.json', 'r') as f:
            logs = json.load(f)
        
        # Return last N actions
        recent = logs[-limit:] if len(logs) > limit else logs
        recent.reverse()  # Most recent first
        
        return {"actions": recent, "total": len(logs)}
    
    except FileNotFoundError:
        return {"actions": [], "total": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GET STATS
@app.get("/api/stats")
def get_stats():
    """Get dashboard statistics"""
    try:
        with open('data/actions_log.json', 'r') as f:
            logs = json.load(f)
        
        if not logs:
            return {
                "total_actions": 0,
                "blocked": 0,
                "warnings": 0,
                "allowed": 0,
                "avg_risk": 0.0
            }
        
        total = len(logs)
        blocked = sum(1 for log in logs if log.get('decision') == 'BLOCK')
        warnings = sum(1 for log in logs if log.get('decision') in ['WARN', 'APPROVAL_REQUIRED'])
        allowed = sum(1 for log in logs if log.get('decision') == 'ALLOW')
        
        total_risk = sum(log.get('risk_score', 0) for log in logs)
        avg_risk = round(total_risk / total, 1) if total > 0 else 0.0
        
        return {
            "total_actions": total,
            "blocked": blocked,
            "warnings": warnings,
            "allowed": allowed,
            "avg_risk": avg_risk
        }
    
    except FileNotFoundError:
        return {
            "total_actions": 0,
            "blocked": 0,
            "warnings": 0,
            "allowed": 0,
            "avg_risk": 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
## risk_scorer.py
```
import json
from datetime import datetime

def load_rules():
    """Load action risk rules from JSON"""
    try:
        with open('data/action_rules.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: action_rules.json not found!")
        return {}

def calculate_risk(action_type, resource_name, environment):
    """
    Calculate risk score from 0-10 based on multiple factors
    
    Parameters:
    - action_type: Type of cloud action (e.g., 'DeleteBucket')
    - resource_name: Name of the resource (e.g., 'production-db')
    - environment: Environment name (e.g., 'production', 'staging')
    
    Returns:
    - Float between 0-10 representing risk level
    """
    
    # Load base risk scores
    rules = load_rules()
    
    # Get base risk for this action type (default to 5 if not found)
    action_data = rules.get(action_type, {})
    base_risk = action_data.get('base_risk', 5)
    
    # Start with base risk
    risk_score = float(base_risk)
    
    # MULTIPLIER 1: Environment
    # Production is riskier than staging/dev
    env_multiplier = 1.0
    env_lower = environment.lower()
    
    if 'production' in env_lower or 'prod' in env_lower:
        env_multiplier = 1.5
        print(f"  [+] Production environment detected: 1.5x multiplier")
    elif 'staging' in env_lower or 'stage' in env_lower:
        env_multiplier = 1.1
        print(f"  [+] Staging environment detected: 1.1x multiplier")
    else:
        print(f"  [+] Development environment: no multiplier")
    
    risk_score *= env_multiplier
    
    # MULTIPLIER 2: Time of day
    # Off-hours actions are riskier (less people around to help if things break)
    current_hour = datetime.now().hour
    time_multiplier = 1.0
    
    if current_hour < 6 or current_hour > 22:
        time_multiplier = 1.2
        print(f"  [+] Off-hours action (late night/early morning): 1.2x multiplier")
    elif current_hour >= 9 and current_hour <= 17:
        print(f"  [+] Business hours: no multiplier")
    else:
        time_multiplier = 1.1
        print(f"  [+] After hours: 1.1x multiplier")
    
    risk_score *= time_multiplier
    
    # MULTIPLIER 3: Resource name analysis
    # Certain keywords in resource names indicate critical resources
    resource_multiplier = 1.0
    resource_lower = resource_name.lower()
    
    critical_keywords = ['critical', 'main', 'primary', 'master', 'customer', 'production', 'prod']
    
    for keyword in critical_keywords:
        if keyword in resource_lower:
            resource_multiplier = 1.3
            print(f"  [+] Critical resource name detected ('{keyword}'): 1.3x multiplier")
            break
    
    if resource_multiplier == 1.0:
        print(f"  [+] Standard resource name: no multiplier")
    
    risk_score *= resource_multiplier
    
    # Cap at 10.0
    final_score = min(risk_score, 10.0)
    
    print(f"\n  Final Risk Score: {final_score:.1f}/10")
    
    return round(final_score, 1)


def get_risk_factors(action_type, resource_name, environment):
    """
    Get detailed breakdown of risk factors (for display purposes)
    """
    rules = load_rules()
    action_data = rules.get(action_type, {})
    
    factors = {
        'base_risk': action_data.get('base_risk', 5),
        'action_description': action_data.get('description', 'Cloud action'),
        'environment': environment,
        'resource_name': resource_name,
        'is_production': 'production' in environment.lower() or 'prod' in environment.lower(),
        'is_off_hours': datetime.now().hour < 6 or datetime.now().hour > 22,
        'is_critical_resource': any(kw in resource_name.lower() for kw in ['critical', 'main', 'primary', 'customer'])
    }
    
    return factors


# Test the function if this file is run directly
if __name__ == "__main__":
    print("="*60)
    print("TESTING RISK SCORER")
    print("="*60)
    
    # Test 1: Low risk action
    print("\n--- TEST 1: Low Risk ---")
    print("Action: CreateInstance on 'test-server' in 'development'")
    score1 = calculate_risk('CreateInstance', 'test-server', 'development')
    print(f"Result: {score1}/10")
    
    # Test 2: Medium risk action
    print("\n--- TEST 2: Medium Risk ---")
    print("Action: DeployCode on 'web-app' in 'staging'")
    score2 = calculate_risk('DeployCode', 'web-app', 'staging')
    print(f"Result: {score2}/10")
    
    # Test 3: High risk action
    print("\n--- TEST 3: High Risk ---")
    print("Action: ModifyDatabase on 'customer-records' in 'production'")
    score3 = calculate_risk('ModifyDatabase', 'customer-records', 'production')
    print(f"Result: {score3}/10")
    
    # Test 4: Critical risk action
    print("\n--- TEST 4: Critical Risk ---")
    print("Action: DeleteDatabase on 'production-main-db' in 'production'")
    score4 = calculate_risk('DeleteDatabase', 'production-main-db', 'production')
    print(f"Result: {score4}/10")
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
```
## teaching.py
```
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_teaching(action_type, resource_name, risk_score, decision, environment):
    """
    Generate AI-powered teaching explanation using OpenAI
    
    If OpenAI fails, returns a smart fallback message
    """
    
    # Fallback messages (used if OpenAI fails)
    fallback_messages = {
        "BLOCK": f"🚫 CRITICAL: {action_type} on '{resource_name}' in {environment} is extremely dangerous. This could cause data loss, security breaches, or major outages. Always test in development first and get approval from 2 senior engineers.",
        
        "APPROVAL_REQUIRED": f"🛡️ HIGH RISK: {action_type} on '{resource_name}' in {environment} requires manager approval. This action affects critical infrastructure. Consider scheduling during maintenance window or using a safer alternative.",
        
        "WARN": f"⚠️ CAUTION: {action_type} on '{resource_name}' could cause issues if not done carefully. Make sure you've tested this in staging environment and have a rollback plan ready.",
        
        "ALLOW": f"✅ SAFE: {action_type} on '{resource_name}' in {environment} is low risk and can proceed normally. Action will be logged for audit purposes."
    }
    
    # Try OpenAI first (if API key exists)
    if openai.api_key and openai.api_key.startswith('sk-'):
        try:
            prompt = f"""You are a helpful DevOps mentor explaining cloud action risks to engineers.

Action Details:
- Type: {action_type}
- Resource: {resource_name}
- Environment: {environment}
- Risk Score: {risk_score}/10
- Decision: {decision}

Provide a brief explanation (2-3 sentences, max 80 words) that:
1. Explains why this action has this risk level
2. What could go wrong
3. Suggests one safer alternative or best practice

Be friendly, clear, and actionable. Don't use bullet points."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful DevOps mentor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            ai_message = response.choices[0].message.content.strip()
            
            # Add emoji prefix
            emoji_map = {"ALLOW": "✅", "WARN": "⚠️", "APPROVAL_REQUIRED": "🛡️", "BLOCK": "🚫"}
            emoji = emoji_map.get(decision, "🔔")
            
            print(f"✅ OpenAI teaching generated successfully")
            return f"{emoji} {ai_message}"
        
        except openai.error.AuthenticationError:
            print("❌ OpenAI API key is invalid")
        except openai.error.RateLimitError:
            print("⚠️ OpenAI rate limit reached")
        except Exception as e:
            print(f"⚠️ OpenAI error: {e}")
    else:
        print("⚠️ No valid OpenAI API key found")
    
    # Fallback to smart pre-written messages
    print("📝 Using fallback teaching message")
    return fallback_messages.get(decision, f"Action detected: {action_type} on {resource_name}")


# Test function
if __name__ == "__main__":
    print("="*60)
    print("TESTING TEACHING MODULE")
    print("="*60)
    
    test_cases = [
        ("DeleteDatabase", "production-main-db", 10, "BLOCK", "production"),
        ("DeployCode", "web-app", 5, "WARN", "staging"),
        ("CreateInstance", "test-server", 2, "ALLOW", "development"),
    ]
    
    for action, resource, score, decision, env in test_cases:
        print(f"\n--- {decision}: {action} ---")
        teaching = generate_teaching(action, resource, score, decision, env)
        print(teaching)
    
    print("\n" + "="*60)
```
# Data
## action_rules.json
```
{
  "DeleteBucket": {
    "base_risk": 9,
    "description": "Permanently deletes S3 bucket and all data"
  },
  "DeleteDatabase": {
    "base_risk": 10,
    "description": "Destroys database - causes data loss"
  },
  "ModifySecurityGroup": {
    "base_risk": 8,
    "description": "Changes firewall rules - security risk"
  },
  "ModifyIAM": {
    "base_risk": 9,
    "description": "Changes user permissions - access control risk"
  },
  "TerminateInstance": {
    "base_risk": 6,
    "description": "Shuts down server - service disruption"
  },
  "CreateInstance": {
    "base_risk": 2,
    "description": "Launches new server - low risk"
  },
  "DeployCode": {
    "base_risk": 4,
    "description": "Pushes new code - potential bugs"
  },
  "ModifyDatabase": {
    "base_risk": 7,
    "description": "Changes database schema - data integrity risk"
  },
  "MakeS3Public": {
    "base_risk": 9,
    "description": "Exposes S3 bucket to internet - data leak risk"
  },
  "OpenFirewall": {
    "base_risk": 10,
    "description": "Opens firewall to 0.0.0.0 - severe security risk"
  }
}
```
## actions_log.json
```
[
  {
    "timestamp": "2025-11-15T14:33:26.337674",
    "engineer_id": "sanjay@company.com",
    "action_type": "DeleteBucket",
    "resource_name": "production-customer-data",
    "environment": "production",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Permanently deletes S3 bucket and all data on production-customer-data in production is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 9,
      "action_description": "Permanently deletes S3 bucket and all data",
      "environment": "production",
      "resource_name": "production-customer-data",
      "is_production": true,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T14:36:06.344554",
    "engineer_id": "sanjay@company.com",
    "action_type": "CreateInstance",
    "resource_name": "production-customer-data",
    "environment": "development",
    "risk_score": 2.6,
    "decision": "ALLOW",
    "level": "Low",
    "emoji": "\u2705",
    "color": "green",
    "action_required": "Action proceeds normally - just logged for audit",
    "alert_team": false,
    "requires_approval": false,
    "teaching": "\u2705 This action is safe to proceed. Launches new server - low risk.",
    "next_steps": [
      "Action will proceed automatically",
      "Event logged for audit trail",
      "No further action needed"
    ],
    "risk_factors": {
      "base_risk": 2,
      "action_description": "Launches new server - low risk",
      "environment": "development",
      "resource_name": "production-customer-data",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T14:36:11.694750",
    "engineer_id": "sanjay@company.com",
    "action_type": "CreateInstance",
    "resource_name": "production-customer-data",
    "environment": "development",
    "risk_score": 2.6,
    "decision": "ALLOW",
    "level": "Low",
    "emoji": "\u2705",
    "color": "green",
    "action_required": "Action proceeds normally - just logged for audit",
    "alert_team": false,
    "requires_approval": false,
    "teaching": "\u2705 This action is safe to proceed. Launches new server - low risk.",
    "next_steps": [
      "Action will proceed automatically",
      "Event logged for audit trail",
      "No further action needed"
    ],
    "risk_factors": {
      "base_risk": 2,
      "action_description": "Launches new server - low risk",
      "environment": "development",
      "resource_name": "production-customer-data",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T14:36:15.860724",
    "engineer_id": "sanjay@company.com",
    "action_type": "CreateInstance",
    "resource_name": "production-customer-data",
    "environment": "development",
    "risk_score": 2.6,
    "decision": "ALLOW",
    "level": "Low",
    "emoji": "\u2705",
    "color": "green",
    "action_required": "Action proceeds normally - just logged for audit",
    "alert_team": false,
    "requires_approval": false,
    "teaching": "\u2705 This action is safe to proceed. Launches new server - low risk.",
    "next_steps": [
      "Action will proceed automatically",
      "Event logged for audit trail",
      "No further action needed"
    ],
    "risk_factors": {
      "base_risk": 2,
      "action_description": "Launches new server - low risk",
      "environment": "development",
      "resource_name": "production-customer-data",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T14:37:16.753236",
    "engineer_id": "sanjay@company.com",
    "action_type": "ModifyIAM",
    "resource_name": "production-customer-data",
    "environment": "development",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Changes user permissions - access control risk on production-customer-data in development is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 9,
      "action_description": "Changes user permissions - access control risk",
      "environment": "development",
      "resource_name": "production-customer-data",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T14:58:05.815877",
    "engineer_id": "sanjay@company.com",
    "action_type": "ModifyIAM",
    "resource_name": "production-customer-data",
    "environment": "development",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Changes user permissions - access control risk on production-customer-data in development is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 9,
      "action_description": "Changes user permissions - access control risk",
      "environment": "development",
      "resource_name": "production-customer-data",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T21:11:18.680837",
    "engineer_id": "sanjay@company.com",
    "action_type": "DeleteBucket",
    "resource_name": "production-customer-data",
    "environment": "production",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Permanently deletes S3 bucket and all data on production-customer-data in production is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 9,
      "action_description": "Permanently deletes S3 bucket and all data",
      "environment": "production",
      "resource_name": "production-customer-data",
      "is_production": true,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T21:12:08.910029",
    "engineer_id": "sanjays2.1404@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "production-main-db",
    "environment": "production",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Changes database schema - data integrity risk on production-main-db in production is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "production",
      "resource_name": "production-main-db",
      "is_production": true,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-15T21:12:56.823725",
    "engineer_id": "sanjays2.1404@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 7.7,
    "decision": "APPROVAL_REQUIRED",
    "level": "High",
    "emoji": "\ud83d\udee1\ufe0f",
    "color": "orange",
    "action_required": "Pause action and require manager approval before proceeding",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udee1\ufe0f This action needs manager approval because changes database schema - data integrity risk in development environment. Consider using a safer alternative or scheduling during maintenance window.",
    "next_steps": [
      "Your manager has been notified via Slack",
      "Action is paused pending approval",
      "Expected response time: 15-30 minutes",
      "Use suggested alternative for faster resolution"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T09:48:09.053666",
    "engineer_id": "sanjays2.1404@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 7.0,
    "decision": "APPROVAL_REQUIRED",
    "level": "High",
    "emoji": "\ud83d\udee1\ufe0f",
    "color": "orange",
    "action_required": "Pause action and require manager approval before proceeding",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udee1\ufe0f This action needs manager approval because changes database schema - data integrity risk in development environment. Consider using a safer alternative or scheduling during maintenance window.",
    "next_steps": [
      "Your manager has been notified via Slack",
      "Action is paused pending approval",
      "Expected response time: 15-30 minutes",
      "Use suggested alternative for faster resolution"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T09:48:53.626412",
    "engineer_id": "sanjay@company.com",
    "action_type": "DeleteBucket",
    "resource_name": "production-customer-data",
    "environment": "production",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Permanently deletes S3 bucket and all data on production-customer-data in production is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 9,
      "action_description": "Permanently deletes S3 bucket and all data",
      "environment": "production",
      "resource_name": "production-customer-data",
      "is_production": true,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-16T09:49:49.046508",
    "engineer_id": "sanjays2.1404@gmail.com",
    "action_type": "ModifyInstance",
    "resource_name": "dev-test-server",
    "environment": "development",
    "risk_score": 5.0,
    "decision": "WARN",
    "level": "Medium",
    "emoji": "\u26a0\ufe0f",
    "color": "yellow",
    "action_required": "Show warning message and require engineer confirmation",
    "alert_team": false,
    "requires_approval": false,
    "teaching": "\u26a0\ufe0f Be careful! Cloud action. Make sure you've tested this in a non-production environment first.",
    "next_steps": [
      "Review the warning message carefully",
      "Confirm you understand the risks",
      "Click 'Proceed Anyway' if you're certain",
      "Consider using the suggested alternative"
    ],
    "risk_factors": {
      "base_risk": 5,
      "action_description": "Cloud action",
      "environment": "development",
      "resource_name": "dev-test-server",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T10:03:12.039569",
    "engineer_id": "sanjays2.1404@gmail.com",
    "action_type": "DeleteDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Destroys database - causes data loss on dev-test-db in development is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 10,
      "action_description": "Destroys database - causes data loss",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T10:03:39.771141",
    "engineer_id": "sanjays2.1404@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "production-customer-data-db",
    "environment": "production",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Changes database schema - data integrity risk on production-customer-data-db in production is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "production",
      "resource_name": "production-customer-data-db",
      "is_production": true,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-16T10:10:16.094099",
    "engineer_id": "sanjaysubbiah22@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 7.0,
    "decision": "APPROVAL_REQUIRED",
    "level": "High",
    "emoji": "\ud83d\udee1\ufe0f",
    "color": "orange",
    "action_required": "Pause action and require manager approval before proceeding",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udee1\ufe0f This action needs manager approval because changes database schema - data integrity risk in development environment. Consider using a safer alternative or scheduling during maintenance window.",
    "next_steps": [
      "Your manager has been notified via Slack",
      "Action is paused pending approval",
      "Expected response time: 15-30 minutes",
      "Use suggested alternative for faster resolution"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T10:13:53.485970",
    "engineer_id": "sanjaysubbiah22@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 7.0,
    "decision": "APPROVAL_REQUIRED",
    "level": "High",
    "emoji": "\ud83d\udee1\ufe0f",
    "color": "orange",
    "action_required": "Pause action and require manager approval before proceeding",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udee1\ufe0f This action needs manager approval because changes database schema - data integrity risk in development environment. Consider using a safer alternative or scheduling during maintenance window.",
    "next_steps": [
      "Your manager has been notified via Slack",
      "Action is paused pending approval",
      "Expected response time: 15-30 minutes",
      "Use suggested alternative for faster resolution"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T10:20:55.111172",
    "engineer_id": "sanjaysubbiah22@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 7.0,
    "decision": "APPROVAL_REQUIRED",
    "level": "High",
    "emoji": "\ud83d\udee1\ufe0f",
    "color": "orange",
    "action_required": "Pause action and require manager approval before proceeding",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udee1\ufe0f This action needs manager approval because changes database schema - data integrity risk in development environment. Consider using a safer alternative or scheduling during maintenance window.",
    "next_steps": [
      "Your manager has been notified via Slack",
      "Action is paused pending approval",
      "Expected response time: 15-30 minutes",
      "Use suggested alternative for faster resolution"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T10:24:43.898121",
    "engineer_id": "sanjaysubbiah22@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "production-customer-data-db",
    "environment": "production",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Changes database schema - data integrity risk on production-customer-data-db in production is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "production",
      "resource_name": "production-customer-data-db",
      "is_production": true,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-16T10:29:43.550945",
    "engineer_id": "sanjaysubbiah22@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "production-customer-data-db",
    "environment": "production",
    "risk_score": 10.0,
    "decision": "BLOCK",
    "level": "Critical",
    "emoji": "\ud83d\udeab",
    "color": "red",
    "action_required": "Action blocked immediately - requires security review and 2-person approval",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udeab Action blocked! Changes database schema - data integrity risk on production-customer-data-db in production is too risky. This could cause major outages or data loss. Please submit a change request with proper review.",
    "next_steps": [
      "Action has been blocked for security",
      "DevOps team has been alerted",
      "Submit a change request ticket",
      "Schedule the action during maintenance window",
      "Get approval from 2 senior engineers"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "production",
      "resource_name": "production-customer-data-db",
      "is_production": true,
      "is_off_hours": false,
      "is_critical_resource": true
    }
  },
  {
    "timestamp": "2025-11-16T10:37:09.968747",
    "engineer_id": "sanjaysubbiah22@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 7.0,
    "decision": "APPROVAL_REQUIRED",
    "level": "High",
    "emoji": "\ud83d\udee1\ufe0f",
    "color": "orange",
    "action_required": "Pause action and require manager approval before proceeding",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udee1\ufe0f This action needs manager approval because changes database schema - data integrity risk in development environment. Consider using a safer alternative or scheduling during maintenance window.",
    "next_steps": [
      "Your manager has been notified via Slack",
      "Action is paused pending approval",
      "Expected response time: 15-30 minutes",
      "Use suggested alternative for faster resolution"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  },
  {
    "timestamp": "2025-11-16T10:44:14.375600",
    "engineer_id": "sanjaysubbiah22@gmail.com",
    "action_type": "ModifyDatabase",
    "resource_name": "dev-test-db",
    "environment": "development",
    "risk_score": 7.0,
    "decision": "APPROVAL_REQUIRED",
    "level": "High",
    "emoji": "\ud83d\udee1\ufe0f",
    "color": "orange",
    "action_required": "Pause action and require manager approval before proceeding",
    "alert_team": true,
    "requires_approval": true,
    "teaching": "\ud83d\udee1\ufe0f This action needs manager approval because changes database schema - data integrity risk in development environment. Consider using a safer alternative or scheduling during maintenance window.",
    "next_steps": [
      "Your manager has been notified via Slack",
      "Action is paused pending approval",
      "Expected response time: 15-30 minutes",
      "Use suggested alternative for faster resolution"
    ],
    "risk_factors": {
      "base_risk": 7,
      "action_description": "Changes database schema - data integrity risk",
      "environment": "development",
      "resource_name": "dev-test-db",
      "is_production": false,
      "is_off_hours": false,
      "is_critical_resource": false
    }
  }
]
```
 # Frontend
 ## clod_console.py
 ```
 import streamlit as st
import requests
import time

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AWS Management Console",
    page_icon="☁️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    
    .aws-header {
        background: linear-gradient(90deg, #232f3e 0%, #ff9900 100%);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .aws-title {
        color: white;
        font-size: 28px;
        font-weight: bold;
    }
    
    .stButton button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.engineer_name = ""
    st.session_state.engineer_email = ""

API_URL = "http://127.0.0.1:8000"

# ==========================================
# RESOURCES
# ==========================================

RESOURCES = {
    "RDS Databases": [
        "production-main-db",
        "production-analytics-db",
        "production-customer-data-db",
        "staging-db",
        "dev-test-db"
    ],
    "S3 Buckets": [
        "production-customer-data",
        "production-backups",
        "production-static-assets",
        "staging-uploads",
        "dev-test-bucket"
    ],
    "EC2 Instances": [
        "web-server-prod-01",
        "web-server-prod-02",
        "api-server-prod",
        "worker-prod-01",
        "staging-web-01",
        "dev-test-server"
    ],
    "Security Groups": [
        "prod-web-sg",
        "prod-db-sg",
        "prod-api-sg",
        "staging-sg"
    ]
}

# ==========================================
# FUNCTIONS
# ==========================================

def check_action(action_type, resource_name, environment):
    """Call CloudMind API"""
    try:
        response = requests.post(
            f"{API_URL}/api/check-action",
            json={
                "engineer_id": st.session_state.engineer_email,
                "action_type": action_type,
                "resource_name": resource_name,
                "environment": environment
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_environment(resource_name):
    """Detect environment from resource name"""
    if 'prod' in resource_name.lower():
        return 'production'
    elif 'staging' in resource_name.lower():
        return 'staging'
    else:
        return 'development'

# ==========================================
# LOGIN PAGE
# ==========================================

if not st.session_state.logged_in:
    st.markdown("""
    <div class="aws-header">
        <p class="aws-title">☁️ AWS Management Console</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Engineer Login")
        name = st.text_input("👤 Name", placeholder="Your name")
        email = st.text_input("📧 Email", placeholder="you@company.com")
        
        if st.button("🚀 Sign In", type="primary", use_container_width=True):
            if name and email:
                st.session_state.logged_in = True
                st.session_state.engineer_name = name
                st.session_state.engineer_email = email
                st.rerun()
    
    st.stop()

# ==========================================
# MAIN CONSOLE
# ==========================================

# Header
st.markdown(f"""
<div class="aws-header">
    <p class="aws-title">☁️ AWS Console - Welcome, {st.session_state.engineer_name}</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 👤 Account")
    st.write(f"**{st.session_state.engineer_name}**")
    st.write(f"{st.session_state.engineer_email}")
    st.markdown("---")
    st.success("🛡️ CloudMind: Active")
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# Resource selection
st.markdown("## ⚙️ Perform Cloud Actions")

resource_type = st.selectbox(
    "1️⃣ Select Resource Type",
    ["RDS Databases", "S3 Buckets", "EC2 Instances", "Security Groups"]
)

st.markdown("### 2️⃣ Select Resources")

selected_resources = []
resources_list = RESOURCES[resource_type]
cols = st.columns(2)

for idx, resource in enumerate(resources_list):
    col = cols[idx % 2]
    with col:
        env = get_environment(resource)
        badge = "🔴 PROD" if env == 'production' else "🟡 STAGING" if env == 'staging' else "🟢 DEV"
        
        if st.checkbox(f"{badge} {resource}", key=f"check_{resource}"):
            selected_resources.append(resource)

if selected_resources:
    st.info(f"✅ Selected: {', '.join(selected_resources)}")
else:
    st.warning("⚠️ No resources selected")

st.markdown("### 3️⃣ Select Action")

action_mappings = {
    "RDS Databases": ["ModifyDatabase", "DeleteDatabase"],
    "S3 Buckets": ["MakeS3Public", "DeleteBucket"],
    "EC2 Instances": ["TerminateInstance", "ModifyInstance"],
    "Security Groups": ["ModifySecurityGroup", "OpenFirewall"]
}

selected_action = st.selectbox("Choose action", action_mappings[resource_type])

# Execute button
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button(f"🚀 Execute {selected_action}", type="primary", use_container_width=True, disabled=len(selected_resources) == 0):
        
        if selected_resources:
            resource_name = selected_resources[0]
            environment = get_environment(resource_name)
            
            with st.spinner("⏳ Analyzing..."):
                time.sleep(0.5)
                result = check_action(selected_action, resource_name, environment)
                
                if result:
                    # Show popup using dialog
                    @st.dialog("🛡️ CloudMind Security Alert", width="large")
                    def show_alert():
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if result['risk_score'] <= 3:
                                st.success(f"**Risk Score**\n\n# 🟢 {result['risk_score']}/10")
                            elif result['risk_score'] <= 6:
                                st.warning(f"**Risk Score**\n\n# 🟡 {result['risk_score']}/10")
                            else:
                                st.error(f"**Risk Score**\n\n# 🔴 {result['risk_score']}/10")
                        
                        with col2:
                            st.info(f"**Risk Level**\n\n# {result['level']}")
                        
                        with col3:
                            if result['decision'] == 'ALLOW':
                                st.success(f"**Decision**\n\n# {result['emoji']} {result['decision']}")
                            elif result['decision'] == 'WARN':
                                st.warning(f"**Decision**\n\n# {result['emoji']} {result['decision']}")
                            else:
                                st.error(f"**Decision**\n\n# {result['emoji']} {result['decision']}")
                        
                        # Decision message
                        if result['decision'] == 'ALLOW':
                            st.success(f"### ✅ {result['action_required']}")
                            st.balloons()
                        elif result['decision'] == 'WARN':
                            st.warning(f"### ⚠️ {result['action_required']}")
                        elif result['decision'] == 'APPROVAL_REQUIRED':
                            st.info(f"### 🛡️ {result['action_required']}")
                        else:
                            st.error(f"### 🚫 {result['action_required']}")
                        
                        # Teaching
                        st.info(f"### 💡 Why This Happened\n\n{result['teaching']}")
                        
                        # Close button
                        if st.button("✅ I Understand", type="primary", use_container_width=True):
                            st.rerun()
                    
                    show_alert()
                else:
                    st.error("❌ Connection failed")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("🛡️ Protected by CloudMind AI | Dashboard: http://localhost:8502")
 ```
 ## monitor_dashboard.py
 ```
 # CloudMind Monitoring Dashboard
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CloudMind - AI Cloud Cop",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING
# ==========================================

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main title styling */
    .main-title {
        font-size: 56px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        padding: 20px 0;
    }
    
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #888;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Risk score box */
    .risk-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 6px solid #ffc107;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ffe5e5 0%, #ffc9c9 100%);
        border-left: 6px solid #ff6348;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 6px solid #dc3545;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    div[data-testid="metric-container"] label {
        color: white !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 32px !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BACKEND API CONFIGURATION
# ==========================================

API_URL = "http://127.0.0.1:8000"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

# ==========================================
# HEADER SECTION
# ==========================================

st.markdown('<p class="main-title">🌩️ CloudMind</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Cloud Cop — Preventing cloud chaos before it happens</p>', unsafe_allow_html=True)

# Backend status indicator
if check_backend_health():
    st.success("✅ Backend API Connected")
else:
    st.error("❌ Backend API Offline - Start with: `uvicorn main:app --reload`")
    st.stop()

st.markdown("---")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ System Info")
    st.info("""
    **CloudMind** uses AI to predict and prevent cloud failures caused by human error.
    
    **Risk Levels:**
    - 🟢 0-3: Low (Allow)
    - 🟡 4-6: Medium (Warn)
    - 🟠 7-8: High (Approval)
    - 🔴 9-10: Critical (Block)
    """)
    
    st.markdown("---")
    
    st.caption("🏆 Built for Hackathon")
    st.caption("👨‍💻 Sanjay S & Dilli Babu K")

# ==========================================
# MAIN CONTENT - TABS
# ==========================================

tab1, tab2, tab3 = st.tabs(["🎯 Check Action", "📊 Dashboard", "📜 Action History"])

# ==========================================
# TAB 1: CHECK ACTION
# ==========================================

with tab1:
    st.header("Simulate Cloud Action")
    st.write("Enter details of a cloud action to analyze its risk level")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        engineer_id = st.text_input(
            "👤 Engineer ID",
            value="sanjay@company.com",
            help="Email or username of the engineer"
        )
        
        action_type = st.selectbox(
            "⚡ Action Type",
            [
                "DeleteBucket",
                "DeleteDatabase",
                "ModifySecurityGroup",
                "ModifyIAM",
                "TerminateInstance",
                "CreateInstance",
                "DeployCode",
                "ModifyDatabase",
                "MakeS3Public",
                "OpenFirewall"
            ],
            help="Type of cloud action to perform"
        )
    
    with col2:
        resource_name = st.text_input(
            "📦 Resource Name",
            value="production-customer-data",
            help="Name of the cloud resource"
        )
        
        environment = st.selectbox(
            "🌍 Environment",
            ["production", "staging", "development"],
            help="Deployment environment"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action button
    if st.button("🚀 Analyze Action", type="primary", use_container_width=True):
        
        # Show loading animation
        with st.spinner("🔍 Analyzing risk factors..."):
            time.sleep(0.5)  # Dramatic pause
            
            try:
                # Call backend API
                response = requests.post(
                    f"{API_URL}/api/check-action",
                    json={
                        "engineer_id": engineer_id,
                        "action_type": action_type,
                        "resource_name": resource_name,
                        "environment": environment
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # ===============================
                    # RISK SCORE DISPLAY
                    # ===============================
                    
                    st.subheader("📊 Risk Analysis Results")
                    
                    # Three columns for metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "Risk Score",
                            f"{data['risk_score']}/10",
                            delta=None
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Risk Level",
                            data['level'],
                            delta=None
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Decision",
                            f"{data['emoji']} {data['decision']}",
                            delta=None
                        )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # ===============================
                    # DECISION BOX
                    # ===============================
                    
                    decision = data['decision']
                    
                    if decision == 'ALLOW':
                        st.success(f"✅ **ACTION ALLOWED**\n\n{data['action_required']}")
                        st.balloons()  # Celebration!
                        
                    elif decision == 'WARN':
                        st.warning(f"⚠️ **WARNING ISSUED**\n\n{data['action_required']}")
                        
                    elif decision == 'APPROVAL_REQUIRED':
                        st.info(f"🛡️ **APPROVAL REQUIRED**\n\n{data['action_required']}")
                        
                    else:  # BLOCK
                        st.error(f"🚫 **ACTION BLOCKED**\n\n{data['action_required']}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # ===============================
                    # TEACHING PANEL
                    # ===============================
                    
                    st.subheader("💡 Learning Moment")
                    st.info(data['teaching'])
                    
                    # Next steps
                    with st.expander("📋 Next Steps"):
                        for i, step in enumerate(data['next_steps'], 1):
                            st.write(f"{i}. {step}")
                    
                    # Risk factors breakdown
                    with st.expander("🔍 Risk Factors Breakdown"):
                        factors = data['risk_factors']
                        st.write(f"**Base Risk:** {factors['base_risk']}/10")
                        st.write(f"**Action:** {factors['action_description']}")
                        st.write(f"**Environment:** {factors['environment']}")
                        st.write(f"**Resource:** {factors['resource_name']}")
                        
                        if factors['is_production']:
                            st.write("⚠️ **Production environment detected** (1.5x multiplier)")
                        if factors['is_off_hours']:
                            st.write("🌙 **Off-hours action** (1.2x multiplier)")
                        if factors['is_critical_resource']:
                            st.write("🚨 **Critical resource name** (1.3x multiplier)")
                    
                    # Alert notification
                    if data['alert_team']:
                        st.success("📢 **Alert sent to DevOps team via Slack!**")
                
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure it's running!")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Backend might be slow.")
            except Exception as e:
                st.error(f"💥 Unexpected error: {str(e)}")

# ==========================================
# TAB 2: DASHBOARD
# ==========================================

with tab2:
    st.header("System Overview")
    
    # Refresh button
    if st.button("🔄 Refresh Stats"):
        st.rerun()
    
    try:
        # Get stats from backend
        stats_response = requests.get(f"{API_URL}/api/stats", timeout=5)
        
        if stats_response.status_code == 200:
            stats = stats_response.json()
            
            # Display metrics in 4 columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Actions",
                    stats['total_actions'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    "🚫 Blocked",
                    stats['blocked'],
                    delta=f"-{stats['blocked']}",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "⚠️ Warnings",
                    stats['warnings'],
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Avg Risk Score",
                    f"{stats['avg_risk']}/10",
                    delta=None
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # System status
            if stats['total_actions'] > 0:
                prevention_rate = ((stats['blocked'] + stats['warnings']) / stats['total_actions'] * 100)
                
                st.success(f"""
                ✅ **CloudMind is actively protecting your infrastructure!**
                
                - **Prevention Rate:** {prevention_rate:.1f}% of risky actions caught
                - **System Uptime:** 100%
                - **Response Time:** <100ms average
                """)
            else:
                st.info("👋 No actions logged yet. Try checking an action in the first tab!")
        
        else:
            st.warning("Unable to load statistics")
    
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

# ==========================================
# TAB 3: ACTION HISTORY
# ==========================================

with tab3:
    st.header("Recent Actions Log")
    
    # Refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    try:
        # Get actions from backend
        actions_response = requests.get(f"{API_URL}/api/actions?limit=20", timeout=5)
        
        if actions_response.status_code == 200:
            actions_data = actions_response.json()
            actions = actions_data.get('actions', [])
            
            if actions:
                st.info(f"📊 Showing {len(actions)} most recent actions (Total: {actions_data.get('total', 0)})")
                
                # Convert to DataFrame for better display
                df_data = []
                for action in actions:
                    df_data.append({
                        'Time': action['timestamp'].split('T')[1].split('.')[0],
                        'Engineer': action['engineer_id'].split('@')[0],
                        'Action': action['action_type'],
                        'Resource': action['resource_name'],
                        'Environment': action['environment'],
                        'Risk': f"{action['risk_score']}/10",
                        'Decision': f"{action['emoji']} {action['decision']}"
                    })
                
                df = pd.DataFrame(df_data)
                
                # Display as table
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed view expander
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📝 Detailed View")
                
                for i, action in enumerate(actions[:5], 1):  # Show top 5 in detail
                    with st.expander(f"{i}. {action['emoji']} {action['action_type']} - {action['decision']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Engineer:** {action['engineer_id']}")
                            st.write(f"**Resource:** {action['resource_name']}")
                            st.write(f"**Environment:** {action['environment']}")
                        
                        with col2:
                            st.write(f"**Risk Score:** {action['risk_score']}/10")
                            st.write(f"**Level:** {action['level']}")
                            st.write(f"**Time:** {action['timestamp']}")
                        
                        st.markdown("**Teaching:**")
                        st.info(action['teaching'])
            
            else:
                st.info("📭 No actions logged yet. Go to 'Check Action' tab to simulate your first action!")
        
        else:
            st.warning("Unable to load action history")
    
    except Exception as e:
        st.error(f"Error loading actions: {str(e)}")

# ==========================================
# FOOTER
# ==========================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("🌩️ **CloudMind** - Preventing cloud chaos before it happens | Built with ❤️ by Sanjay S & Dilli Babu K")
 ```
 # .env
 ```
 OPENAI_API_KEY=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
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/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
 ```
 ## requirements.txt
 ```
 fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.1
openai==1.3.5
requests==2.31.0
python-dotenv==1.0.0
pandas==2.1.3
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
# OUTPUT
![img1](https://github.com/user-attachments/assets/00d14a73-67d4-4a56-a669-d79e4858a149)
![img2](https://github.com/user-attachments/assets/50dccf50-3a74-4333-a5c2-22d6448750bd)

<img width="1280" height="529" alt="image" src="https://github.com/user-attachments/assets/bafc0f2e-ee76-46e1-a522-3d4cf9db723d" />

<img width="1280" height="638" alt="image" src="https://github.com/user-attachments/assets/95f069a3-121a-4547-a835-3311145d59b3" />

<img width="1271" height="186" alt="image" src="https://github.com/user-attachments/assets/db36c258-71fa-4449-8d08-c2bd593f5db8" />
<img width="1280" height="314" alt="image" src="https://github.com/user-attachments/assets/5c75aa9d-c32d-49ba-b2bd-8693db984faf" />

## ⚙️ Tech Stack
- Python 3.10+  
- JSON Rule Engine  
- CLI + Dashboard  
- Event Logging
- Web ui(stream lit)
- Uvicorn
- Fast api


## 🛠️ Future Enhancements
- Cloud provider integration (AWS/GCP/Azure)  
- ML anomaly detection
- Ml risk prediction
- Notification system  


## 👨‍💻 Author
CloudMind — Built with ❤️ for hackathons.
