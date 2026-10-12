from app.services.llm_interface import generate_explanation
from app.services.memorymanager import memory_manager

sid = memory_manager.create_session()

report = """
Risk Score: 90
Verdict: Likely Phishing
Indicators:
- IP-based URL detected
- Credential request detected
- Urgency language detected
"""

explanation = generate_explanation(sid, report)
print("LLM Explanation:\n", explanation)