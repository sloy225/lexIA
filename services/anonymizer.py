from __future__ import annotations

import re
from dataclasses import dataclass, field

from services.azure_openai import azure_openai_service
from utils.text_utils import truncate_text

_SYSTEM_PROMPT = """\
Tu es un assistant spécialisé dans la détection et l'anonymisation des données personnelles \
dans les documents juridiques français (RGPD / GDPR).

Réponds TOUJOURS en français.
"""

_DETECT_PII_PROMPT = """\
Identifie toutes les données personnelles (PII) dans le texte suivant. \
Classe-les par catégorie et fournis une liste JSON.

Catégories à détecter :
- Noms de personnes physiques
- Adresses (postales, email)
- Numéros de téléphone
- Numéros d'identification (SIRET, SIREN, numéro de sécu, RCS, etc.)
- Dates de naissance
- Coordonnées bancaires (IBAN, etc.)
- Autres données sensibles

Format JSON :
{{
  "pii_found": [
    {{
      "category": "Nom de personne",
      "value": "valeur trouvée dans le texte",
      "replacement": "remplacement suggéré (ex: [PRÉNOM NOM])"
    }}
  ],
  "total_count": 5
}}

Texte :
{text}
"""


@dataclass
class PIIEntity:
    category: str
    value: str
    replacement: str


@dataclass
class AnonymizationResult:
    anonymized_text: str
    entities_found: list[PIIEntity] = field(default_factory=list)
    total_replaced: int = 0


class AnonymizerService:
    # Regex-based pre-anonymization for common French PII patterns
    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
    _PHONE_RE = re.compile(r"\b(?:\+33|0033|0)\s*[1-9](?:[\s.\-]?\d{2}){4}\b")
    _IBAN_RE = re.compile(r"\bFR\d{2}[\s]?(?:\d{4}[\s]?){5}\d{3}\b")
    _SIRET_RE = re.compile(r"\b\d{3}\s?\d{3}\s?\d{3}\s?\d{5}\b")
    _SIREN_RE = re.compile(r"\b\d{3}\s?\d{3}\s?\d{3}\b")

    def detect_pii(self, text: str) -> list[PIIEntity]:
        """Detect PII using Azure OpenAI (AI-based detection)."""
        truncated = truncate_text(text, max_tokens=6000)
        raw = azure_openai_service.chat_with_system(
            system_prompt=_SYSTEM_PROMPT,
            user_message=_DETECT_PII_PROMPT.format(text=truncated),
            temperature=0.0,
            max_tokens=1500,
        )
        return self._parse_pii_response(raw)

    def anonymize_regex(self, text: str) -> tuple[str, int]:
        """
        Fast regex-based anonymization for common patterns.
        Returns (anonymized_text, count_replaced).
        """
        count = 0

        def replace_and_count(pattern: re.Pattern, replacement: str, t: str) -> str:
            nonlocal count
            result, n = pattern.subn(replacement, t)
            count += n
            return result

        text = replace_and_count(self._EMAIL_RE, "[ADRESSE EMAIL]", text)
        text = replace_and_count(self._PHONE_RE, "[NUMÉRO DE TÉLÉPHONE]", text)
        text = replace_and_count(self._IBAN_RE, "[IBAN]", text)
        text = replace_and_count(self._SIRET_RE, "[SIRET]", text)

        return text, count

    def anonymize_full(self, text: str) -> AnonymizationResult:
        """
        Full anonymization: regex first, then AI-based for names and addresses.
        """
        # Step 1: regex-based
        anonymized, regex_count = self.anonymize_regex(text)

        # Step 2: AI-based PII detection
        entities = self.detect_pii(anonymized)

        # Step 3: replace detected entities
        ai_count = 0
        for entity in entities:
            if entity.value and entity.replacement and entity.value in anonymized:
                anonymized = anonymized.replace(entity.value, entity.replacement)
                ai_count += 1

        return AnonymizationResult(
            anonymized_text=anonymized,
            entities_found=entities,
            total_replaced=regex_count + ai_count,
        )

    def _parse_pii_response(self, raw: str) -> list[PIIEntity]:
        import json
        import re as re_module

        json_match = re_module.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            return []
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        entities = []
        for item in data.get("pii_found", []):
            entities.append(
                PIIEntity(
                    category=item.get("category", ""),
                    value=item.get("value", ""),
                    replacement=item.get("replacement", "[DONNÉES ANONYMISÉES]"),
                )
            )
        return entities


anonymizer_service = AnonymizerService()
