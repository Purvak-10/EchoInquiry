HYPOTHESIS_GENERATION_PROMPT = """
You are a rigorous scientific reasoning engine.

Task: Generate falsifiable hypotheses using Popperian falsificationism.

Core Question:
{core_question}

Domain:
{domain}

Sub-questions:
{sub_questions}

Instructions:
- Generate EXACTLY {hypothesis_count} competing hypotheses
- Hypotheses MUST be mutually competing where possible
- AT LEAST ONE must be a NULL hypothesis (no effect / no relationship)
- Each hypothesis MUST be falsifiable
- Clearly describe:
  - mechanism (why it works)
  - predicted evidence (what we expect to observe if true)
  - falsification criteria (what would prove it wrong)
- Assign a reasonable prior confidence (0-1)

STRICT OUTPUT FORMAT:
Return ONLY valid JSON. No explanation.

{{
  "hypotheses": [
    {{
      "id": "h1",
      "statement": "...",
      "mechanism": "...",
      "predicted_evidence": "...",
      "falsification_criteria": "...",
      "confidence_prior": 0.5
    }}
  ]
}}
"""


HYPOTHESIS_EVALUATION_PROMPT = """
You are a scientific evaluator.

Evaluate hypotheses objectively using provided evidence.

Hypotheses:
{hypothesis_json}

Evidence:
{source_evidence_json}

Instructions:
- Evaluate EACH hypothesis independently
- Be skeptical and unbiased
- Prefer falsification when strong opposing evidence exists
- Use ONLY provided evidence (no assumptions)

Return ONLY JSON:

{{
  "evaluations": [
    {{
      "id": "h1",
      "status": "supported|falsified|partially_supported|insufficient_evidence",
      "confidence_posterior": 0.0,
      "supporting_evidence": ["..."],
      "opposing_evidence": ["..."],
      "verdict": "one sentence conclusion"
    }}
  ]
}}
"""
