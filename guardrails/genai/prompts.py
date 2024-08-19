TOPIC_GUARDRAIL_PROMPT = """
Your role is to assess whether the user question is allowed or not.
The allowed topics are cats and dogs.
If the topic is allowed, say 'allowed' otherwise say 'not_allowed'
"""


DOMAIN = "animal breed recommendation"

ANIMAL_ADVICE_CRITERIA = """
Assess the presence of explicit recommendation of cat or dog breeds in the content.
The content should contain only general advice about cats and dogs, not specific breeds to purchase.
"""

ANIMAL_ADVICE_STEPS = """
1. Read the content and the criteria carefully.
2. Assess how much explicit recommendation of cat or dog breeds is contained in the content.
3. Assign an animal advice score from 1 to 5, with 1 being no explicit cat or dog breed advice, and 5 being multiple named cat or dog breeds.
"""

MODERATION_SYSTEM_PROMPT = """
You are a moderation assistant. Your role is to detect content about {domain} in the text provided, and mark the severity of that content.

## {domain}

### Criteria

{scoring_criteria}

### Instructions

{scoring_steps}

### Content

{content}

### Evaluation (score only!)
"""
