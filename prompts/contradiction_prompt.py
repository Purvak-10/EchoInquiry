CLAIM_EXTRACTION_PROMPT = """
You are an expert research analyst.

Extract up to 10 precise, verifiable claims from the source.

SOURCE TITLE:
{source_title}

SOURCE CONTENT:
{source_content}

RULES:
- Extract ONLY specific claims (no vague statements)
- Prefer factual, testable assertions
- Each claim must stand independently
- Avoid duplicates
- If no valid claims exist, return empty list

RETURN ONLY VALID JSON:
{{
  "claims": [
    {{
      "claim_id": "string",
      "claim_text": "string",
      "claim_type": "quantitative|qualitative|causal|comparative",
      "key_terms": ["string"],
      "confidence": 0.0
    }}
  ]
}}
"""


CONTRADICTION_ANALYSIS_PROMPT = """
You are a scientific contradiction detection system.

Analyze whether the two claims contradict each other.

CLAIM A:
{claim_a}
SOURCE A:
{source_a_title}

CLAIM B:
{claim_b}
SOURCE B:
{source_b_title}

DEFINITION OF CONTRADICTION:
- Direct: Opposite conclusions
- Partial: Overlapping but conflicting details
- Methodological: Different methods lead to conflicting results
- Contextual: Conflict due to context differences
- None: No contradiction

RETURN ONLY VALID JSON:
{{
  "is_contradiction": true,
  "severity": "high|medium|low|none",
  "contradiction_type": "direct|partial|methodological|contextual|none",
  "explanation": "string",
  "resolution_hint": "string"
}}
"""
