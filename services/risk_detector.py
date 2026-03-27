from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from services.azure_openai import azure_openai_service
from utils.text_utils import truncate_text

class RiskLevel(str, Enum):
    HIGH = "Élevé"
    MEDIUM = "Moyen"
    LOW = "Faible"


@dataclass
class RiskClause:
    title: str
    excerpt: str
    risk_level: RiskLevel
    explanation: str
    recommendation: str


@dataclass
class RiskReport:
    risks: list[RiskClause] = field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.LOW
    overall_comment: str = ""

    @property
    def high_count(self) -> int:
        return sum(1 for r in self.risks if r.risk_level == RiskLevel.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for r in self.risks if r.risk_level == RiskLevel.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for r in self.risks if r.risk_level == RiskLevel.LOW)


_SYSTEM_PROMPT = """\
Tu es un expert juridique spécialisé dans l'analyse des risques contractuels pour les PME, \
startups et équipes non juridiques françaises. Ton rôle est d'identifier les clauses \
potentiellement risquées ou déséquilibrées dans les contrats.

Réponds TOUJOURS en français, de façon claire et accessible.
"""

_RISK_PROMPT = """\
Analyse le contrat suivant et identifie les clauses à risque. \
Pour chaque clause problématique, fournis une réponse JSON structurée.

Types de risques à rechercher :
- Clauses pénales ou d'indemnisation excessives
- Limitations de responsabilité déséquilibrées
- Clauses de non-concurrence ou d'exclusivité abusives
- Obligations de résultat déguisées en obligations de moyens (ou inverse)
- Conditions de résiliation défavorables
- Renouvellement automatique sans préavis clair
- Cession de droits de propriété intellectuelle trop large
- Clauses attributives de juridiction défavorables
- Pénalités de retard asymétriques
- Confidentialité trop restrictive ou trop permissive

Format de réponse JSON :
{{
  "risks": [
    {{
      "title": "Nom court de la clause",
      "excerpt": "Citation exacte ou résumé de la clause problématique (max 200 caractères)",
      "risk_level": "Élevé|Moyen|Faible",
      "explanation": "Pourquoi cette clause est risquée (1-2 phrases)",
      "recommendation": "Ce que vous devriez faire ou négocier (1-2 phrases)"
    }}
  ],
  "overall_risk": "Élevé|Moyen|Faible",
  "overall_comment": "Évaluation globale du contrat en 1-2 phrases"
}}

Contrat :
{text}
"""


class RiskDetectorService:
    def detect(self, full_text: str) -> RiskReport:
        """Analyse a contract and return a structured risk report."""
        truncated = truncate_text(full_text, max_tokens=12_000)

        raw = azure_openai_service.chat_with_system(
            system_prompt=_SYSTEM_PROMPT,
            user_message=_RISK_PROMPT.format(text=truncated),
            temperature=0.1,
            max_tokens=2500,
        )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> RiskReport:
        import json
        import re

        # Extract JSON block from the response (model may wrap it in markdown)
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            return RiskReport(
                overall_comment="Impossible d'analyser le contrat. Veuillez réessayer.",
                overall_risk=RiskLevel.MEDIUM,
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return RiskReport(
                overall_comment="Erreur lors de l'analyse. Veuillez réessayer.",
                overall_risk=RiskLevel.MEDIUM,
            )

        risks = []
        for r in data.get("risks", []):
            level_str = r.get("risk_level", "Faible")
            level = self._parse_level(level_str)
            risks.append(
                RiskClause(
                    title=r.get("title", ""),
                    excerpt=r.get("excerpt", ""),
                    risk_level=level,
                    explanation=r.get("explanation", ""),
                    recommendation=r.get("recommendation", ""),
                )
            )

        overall = self._parse_level(data.get("overall_risk", "Faible"))

        return RiskReport(
            risks=risks,
            overall_risk=overall,
            overall_comment=data.get("overall_comment", ""),
        )

    def _parse_level(self, value: str) -> RiskLevel:
        mapping = {
            "élevé": RiskLevel.HIGH,
            "eleve": RiskLevel.HIGH,
            "moyen": RiskLevel.MEDIUM,
            "faible": RiskLevel.LOW,
        }
        return mapping.get(value.lower().strip(), RiskLevel.LOW)


risk_detector_service = RiskDetectorService()
