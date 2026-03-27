from __future__ import annotations

from dataclasses import dataclass, field

from services.azure_openai import azure_openai_service
from utils.text_utils import truncate_text

_SYSTEM_PROMPT = """\
Tu es un assistant juridique expert en analyse de contrats. Tu aides des équipes non juridiques \
(RH, commerciaux, achats, startups) à comprendre les contrats en langage clair et accessible.

Règles impératives :
- Réponds TOUJOURS en français.
- Utilise un langage simple, sans jargon juridique inutile.
- Sois factuel et précis.
- Rappelle que tu es un outil d'aide à la compréhension, pas un substitut à un avocat.
"""

_SUMMARY_PROMPT = """\
Analyse le contrat suivant et fournis un résumé structuré en français comprenant :

1. **Type de contrat** : identifie le type (prestation, NDA, travail, CGV/CGU, autre)
2. **Parties impliquées** : qui sont les signataires ?
3. **Objet du contrat** : que couvre ce contrat en 2-3 phrases ?
4. **Durée et dates clés** : date de début, durée, renouvellement, préavis
5. **Obligations principales** : 3-5 obligations essentielles de chaque partie
6. **Conditions financières** : montants, modalités de paiement, pénalités
7. **Points de vigilance** : 2-3 éléments importants à retenir

Contrat :
{text}
"""

_SHORT_SUMMARY_PROMPT = """\
Résume ce contrat en 3 phrases maximum, en langage très simple, \
comme si tu l'expliquais à quelqu'un sans formation juridique.

Contrat :
{text}
"""


@dataclass
class ContractSummary:
    detailed: str
    short: str
    contract_type_detected: str = "Autre"


class SummarizerService:
    def summarize(self, full_text: str, stream: bool = False) -> ContractSummary | object:
        """
        Generate both a detailed structured summary and a short plain-language summary.
        If stream=True, returns a generator for the detailed summary only.
        """
        # Limit input to ~12 000 tokens (leave room for output)
        truncated = truncate_text(full_text, max_tokens=12_000)

        if stream:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _SUMMARY_PROMPT.format(text=truncated)},
            ]
            return azure_openai_service.chat(
                messages, temperature=0.1, max_tokens=2048, stream=True
            )

        # Detailed summary
        detailed = azure_openai_service.chat_with_system(
            system_prompt=_SYSTEM_PROMPT,
            user_message=_SUMMARY_PROMPT.format(text=truncated),
            temperature=0.1,
            max_tokens=2048,
        )

        # Short summary (uses a smaller portion of the text)
        short_text = truncate_text(full_text, max_tokens=4000)
        short = azure_openai_service.chat_with_system(
            system_prompt=_SYSTEM_PROMPT,
            user_message=_SHORT_SUMMARY_PROMPT.format(text=short_text),
            temperature=0.1,
            max_tokens=300,
        )

        # Detect contract type from detailed summary
        contract_type = self._detect_contract_type(detailed)

        return ContractSummary(detailed=detailed, short=short, contract_type_detected=contract_type)

    def _detect_contract_type(self, summary_text: str) -> str:
        type_map = {
            "NDA": "NDA / Accord de confidentialité",
            "confidentialité": "NDA / Accord de confidentialité",
            "travail": "Contrat de travail",
            "emploi": "Contrat de travail",
            "prestation": "Contrat de prestation de services",
            "services": "Contrat de prestation de services",
            "CGV": "CGV / CGU",
            "CGU": "CGV / CGU",
            "conditions générales": "CGV / CGU",
        }
        lower = summary_text.lower()
        for keyword, contract_type in type_map.items():
            if keyword.lower() in lower:
                return contract_type
        return "Autre"


summarizer_service = SummarizerService()
