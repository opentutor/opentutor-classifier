#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
from dataclasses import dataclass
import re

REGEX_METACOGNITIVE = r"\b(idk|belie\w*|don\w*|comprehend\w*|confuse\w*|guess\w*|(?<=n[o']t)\s?\b(know\w*|underst\w*|follow\w*|recogniz\w*|sure\w*|get)\b|messed|no\s?(idea|clue)|lost|forg[eo]t|need\s?help|imagined?|interpret(ed)?|(seen?|saw)|suppos(ed)?)\b"
REGEX_PROFANITY = r"\b(\w*fuck\w*|ass|ass(hole|wipe|wad|kisser)|hell|shit|piss\w*|cock|cock(sucker|head|eater)|douche\w*|bitch\w*|retard[ed]|midget\w*|dyke|fag|faggot|cunt\w*|\w*nigg\w*|tranny|slut\w*|cum[bucket]|dick\w*|pussy\w*|dildo\w*|idiot\w*|(hate you)|stupid\w*)\b"


@dataclass
class SpeechActClassifierResult:
    evaluation: str = ""
    score: float = 0.0


class SpeechActClassifier:
    def check_meta_cognitive(self, result):
        input_sentence = result.input.input_sentence
        return (
            SpeechActClassifierResult(evaluation="Good", score=1)
            if re.search(REGEX_METACOGNITIVE, input_sentence, re.IGNORECASE)
            else SpeechActClassifierResult(evaluation="Bad", score=0)
        )

    def check_profanity(self, result):
        input_sentence = result.input.input_sentence
        return (
            SpeechActClassifierResult(evaluation="Good", score=1)
            if re.search(REGEX_PROFANITY, input_sentence, re.IGNORECASE)
            else SpeechActClassifierResult(evaluation="Bad", score=0)
        )
