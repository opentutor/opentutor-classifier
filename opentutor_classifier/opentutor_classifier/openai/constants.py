SYSTEM_ASSIGNMENT = 'The user provided a list of one or more "answers" to a tutoring question. The answers are provided in JSON format. You are a tutor who is evaluating if the answer is sufficient to show that the user knows a one or more "concepts" which will be labeled "concept_1" to "concept_N". For each answer you evaluate, you must also express how confident you are in your evaluation of how well the answer shows knowledge of each concept. You will also provide a brief justification.'
CONCEPT_HEADER = "These are the concepts for this lesson."
ANSWER_HEADER = "These are my answers provided in JSON to be evaluated."
ANSWER_TEMPLATE = {
    "Please respond in the following format": {
        "answers": {
            "answer_N": {
                "answer text": "string // State the text of the particular answer being classified.",
                "concepts": {
                    "concept_N": {
                        "is_known": "string // true or false. If the input answer implies that the concept is known, the classification should be true. Otherwise it should be false.",
                        "confidence": "float // A 0 to 1 score indicating certainty that a classification is correct. Confidence scores closer to 1 represent higher certainty, and confidence scores closer to 0 represent lower certainty.",
                        "justification": "string // Why you believe the user answer is or is not sufficient to determine if they know the concepts.",
                    }
                },
            }
        }
    }
}
USER_GUARDRAILS = "Only respond with the JSON output in the exact format of the template and no other words or symbols. The output must be valid JSON. Check that the output is valid JSON. \n\n"

OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_DEFAULT_TEMP = 0.1
