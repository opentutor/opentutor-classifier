#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

SYSTEM_ASSIGNMENT = 'The user provided a list of one or more "answers" to a tutoring question. The answers are provided in JSON format. You are a tutor who is evaluating if the answer is sufficient to show that the user knows a one or more "concepts" which will be labeled "concept_1" to "concept_N". For each answer you evaluate, you must also express how confident you are in your evaluation of how well the answer shows knowledge of each concept. You will also provide a brief justification.'
CONCEPT_HEADER = "These are the concepts for this lesson."
ANSWER_HEADER = "These are my answers provided in JSON to be evaluated."
ANSWER_TEMPLATE = {
    "Please respond in the following format": {
        "answers": {
            "answer_N": {
                "answer_text": "string // State the text of the particular answer being classified.",
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
USER_GROUNDTRUTH = "Here are some examples that have already been labeled. They are presented in JSON format, where the answer is given, followed by each concept and a true or false label. Consider these to be ground truth examples.\n"
OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_MODEL = "gpt-3.5-turbo-16k"  # TODO: use tiktoken to estimate token cost and use appropriate model
OPENAI_DEFAULT_TEMP = 0.1
GROUNDTRUTH_FILENAME = "groundtruth.json"
