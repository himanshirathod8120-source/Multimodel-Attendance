"""
recommendation.py
=================
Generates AI-driven teaching suggestions based on engagement patterns.
Uses local Ollama LLM (llama3/mistral) with rule-based fallback.
"""

import time
import logging
import requests
import json
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based fallback suggestions
# ---------------------------------------------------------------------------
_RULE_SUGGESTIONS = {
    "very_low": [
        "⚠️ Class engagement is critically low. Consider taking a 5-minute break.",
        "⚠️ Most students appear disengaged. Try an interactive activity or quick quiz.",
        "⚠️ Energy levels are very low. Consider a physical energizer exercise.",
        "⚠️ Low engagement detected. Ask students an open-ended question to re-focus.",
    ],
    "low": [
        "📉 Engagement is below average. Consider switching to a group discussion format.",
        "📉 Some students seem distracted. Try a think-pair-share activity.",
        "📉 Pacing may be too fast or slow. Check in with students on comprehension.",
        "📉 Consider using a visual aid or demonstration to re-capture attention.",
    ],
    "moderate": [
        "👍 Engagement is moderate. A quick recap question could boost participation.",
        "👍 Consider encouraging quieter students to contribute.",
        "👍 Good progress! A short collaborative task could maintain momentum.",
    ],
    "high": [
        "🌟 Excellent engagement! Students are highly attentive — great pacing.",
        "🌟 Class is very engaged. This is a good time to introduce complex concepts.",
        "🌟 High participation detected. Consider extending the current discussion.",
    ],
}


class RecommendationEngine:
    """
    Generates teaching suggestions using Ollama LLM or rule-based fallback.
    """

    def __init__(self, config: dict):
        self.provider = config.get("provider", "rule_based")
        self.ollama_model = config.get("ollama_model", "llama3.2")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.suggestion_interval = config.get("suggestion_interval_seconds", 120)
        self.alert_threshold = config.get("engagement_alert_threshold", 40)

        self._last_suggestion_time = 0.0
        self._suggestion_counter = 0
        self._ollama_available = False
        self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama service is running."""
        if self.provider != "ollama":
            return
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if resp.status_code == 200:
                self._ollama_available = True
                logger.info(f"Ollama available at {self.ollama_url} (model: {self.ollama_model})")
            else:
                logger.warning("Ollama not responding. Using rule-based fallback.")
        except Exception:
            logger.warning("Ollama not available. Using rule-based fallback.")

    def should_generate(self) -> bool:
        """Check if it's time to generate a new suggestion."""
        return time.time() - self._last_suggestion_time >= self.suggestion_interval

    def generate(
        self,
        class_engagement: float,
        student_averages: dict[str, float],
        anomalies: list[dict],
        session_duration_min: float,
    ) -> Optional[str]:
        """
        Generate a context-aware teaching suggestion.

        Args:
            class_engagement: Current class-wide score (0-100)
            student_averages: {name: avg_score}
            anomalies: List of anomaly dicts from engagement engine
            session_duration_min: Minutes elapsed in session

        Returns:
            Suggestion string or None if not time yet
        """
        if not self.should_generate():
            return None

        self._last_suggestion_time = time.time()
        self._suggestion_counter += 1

        if self._ollama_available and self.provider == "ollama":
            suggestion = self._ollama_suggestion(
                class_engagement, student_averages, anomalies, session_duration_min
            )
            if suggestion:
                return suggestion

        # Fallback to rule-based
        return self._rule_based_suggestion(class_engagement, anomalies)

    def _ollama_suggestion(
        self,
        class_engagement: float,
        student_averages: dict,
        anomalies: list,
        session_duration_min: float,
    ) -> Optional[str]:
        """Generate suggestion using Ollama LLM."""
        low_students = [f"{k} ({v:.0f}%)" for k, v in student_averages.items() if v < 40]
        anomaly_desc = "; ".join([a["alert"] for a in anomalies]) if anomalies else "None"

        prompt = (
            f"You are an AI classroom assistant helping a teacher improve their lesson.\n\n"
            f"Current classroom status:\n"
            f"- Session duration: {session_duration_min:.0f} minutes\n"
            f"- Overall class engagement score: {class_engagement:.0f}/100\n"
            f"- Students with low engagement: {', '.join(low_students) if low_students else 'None'}\n"
            f"- Alerts: {anomaly_desc}\n\n"
            f"Based on this data, provide ONE specific, practical teaching suggestion "
            f"the instructor can immediately act on. Keep it under 3 sentences. "
            f"Be direct and actionable. Do not use markdown."
        )

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 150},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                if text:
                    logger.info(f"Ollama suggestion generated ({len(text)} chars)")
                    return f"🤖 AI Suggestion: {text}"
        except Exception as e:
            logger.warning(f"Ollama request failed: {e}")

        return None

    def _rule_based_suggestion(
        self, class_engagement: float, anomalies: list
    ) -> str:
        """Pick a rule-based suggestion based on engagement level."""
        import random

        if class_engagement < 25:
            key = "very_low"
        elif class_engagement < 45:
            key = "low"
        elif class_engagement < 70:
            key = "moderate"
        else:
            key = "high"

        suggestion = random.choice(_RULE_SUGGESTIONS[key])

        if anomalies:
            names = [a["student"] for a in anomalies[:2]]
            suffix = f" (Students needing attention: {', '.join(names)})"
            suggestion += suffix

        return suggestion

    def get_suggestion_count(self) -> int:
        return self._suggestion_counter
