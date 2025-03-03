from enum import Enum
import html.entities
from piraye import NormalizerBuilder
import json
import html
import re
import os


class ValidationStatus(Enum):
    VALID = -1
    UNKNOWN = 0
    EMPTY_TEXT = 1
    INVALID_CHARS = 2
    DIALOGUE = 3
    INVALID_START_END = 4


class Response:
    def __init__(self, 
                 text: str = None,
                 is_valid: bool = None,
                 status=ValidationStatus.VALID
                 ):
        self.text = text
        self.is_valid = is_valid
        self.status = status

    @classmethod
    def invalid(cls, text: str, status=ValidationStatus.UNKNOWN):
        return cls(text, False, status)

    @classmethod
    def valid(cls, text: str):
        return cls(text, True, ValidationStatus.VALID)

    def __repr__(self):
        if not self.is_valid:
            return 'Invalid Text'
        return self.text


class TextNormalizer:
    def __init__(self):
        self.piraye_normalizer = (
            NormalizerBuilder()
            .alphabet_fa()
            .digit_en()
            .diacritic_delete()
            .punctuation_fa()
            .build()
        )

        self._piraye_remove_extra_spaces = (
            NormalizerBuilder()
            .remove_extra_spaces()
            .build()
        )

        # Load valid characters once during initialization
        self.valid_chars = set(['\u200c', ' ', '?', '؟'])
        
        pwd = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(pwd, 'assets/fa_normal_chars.json'), 'r') as f:
            data = json.load(f)
            for a in data:
                self.valid_chars.add(a['map']['alphabet_fa']['char'])
        with open(os.path.join(pwd, 'assets/fa_puncs.json'), 'r') as f:
            data = json.load(f)
            for punc in data:
                self.valid_chars.add(punc['map']['punc_fa']['char'])
        with open(os.path.join(pwd, 'assets/digits.json'), 'r') as f:
            data = json.load(f)
            for punc in data:
                self.valid_chars.add(punc['map']['digit_en']['char'])

        # Pre-compile regex patterns
        self._star_explanation_pattern = re.compile(r'\*.*\*')
        self._braces_explanation_pattern = re.compile(r'\[.*\]')
        self._prentices_explanation_pattern = re.compile(r'\(.*\)')
        self._quote_explanation_pattern = re.compile(r'".*"')

        self._repeated_question_mark_fa_pattern = re.compile(r'؟{2,}')
        self._repeated_question_mark_pattern = re.compile(r'\?{2,}')
        self._repeated_exclamation_mark_pattern = re.compile(r'\!{2,}')

    def _validate_text(self, text: str) -> Response:
        if not text.strip():
            return Response.invalid(text, ValidationStatus.EMPTY_TEXT)
        elif any(char not in self.valid_chars for char in text):
            return Response.invalid(text, ValidationStatus.INVALID_CHARS)
        else:
            return Response.valid(text)

    def _replace_invalid_texts(self, text: str) -> str:
        replace_list = {
            r'(\.\s*)+': '.',
            r'\n': ' '
        }
        for pattern, replacement in replace_list.items():
            text = re.sub(pattern, replacement, text)
        return text

    def _piraye_normalize(self, text: str) -> str:
        """Applies Piraye library normalization."""
        return self.piraye_normalizer.normalize(text)
    
    def _remove_extra_spaces(self, text: str) -> str:
        """Removes extra spaces."""
        return self._piraye_remove_extra_spaces.normalize(text)

    def _remove_explanations(self, text: str) -> str:
        """Removes inline explanations like *...*, [...], (...), "..." """
        text = self._star_explanation_pattern.sub('', text)
        text = self._braces_explanation_pattern.sub('', text)
        text = self._prentices_explanation_pattern.sub('', text)
        text = self._quote_explanation_pattern.sub('', text)
        return text

    def _remove_repeated_signs(self, text: str) -> str:
        """Removes consecutive punctuation marks."""
        text = self._repeated_question_mark_fa_pattern.sub('؟', text)
        text = self._repeated_question_mark_pattern.sub('?', text)
        text = self._repeated_exclamation_mark_pattern.sub('!', text)
        return text

    def _convert_html_entities(self, text: str) -> str:
        """Converts HTML entities to normal characters."""
        return html.unescape(text)

    def normalize(self, text: str):
        """Runs all normalization steps and returns a response."""
        text = self._convert_html_entities(text)
        text = self._piraye_normalize(text)
        text = self._remove_explanations(text)
        text = self._remove_repeated_signs(text)
        text = self._replace_invalid_texts(text)
        text = self._remove_extra_spaces(text)
        return self._validate_text(text)