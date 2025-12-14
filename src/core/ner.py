import spacy
from spacy.matcher import Matcher

from src.utils import config


class NERService:
    def __init__(self):
        self.nlp = spacy.load(config.NER_MODEL_NAME)

        self.matcher = Matcher(self.nlp.vocab)

        self.pattern_adj_noun = [
            {"POS": "ADJ"},
            {"POS": {"IN": ["NOUN", "PROPN"]}}
        ]

        self.pattern_noun_noun = [
            {"POS": {"IN": ["NOUN", "PROPN"]}},
            {"POS": {"IN": ["NOUN", "PROPN"]}}
        ]

        self.matcher.add("Adj_Noun", [self.pattern_adj_noun])
        self.matcher.add("Noun_Noun", [self.pattern_noun_noun])


    def extract_search_terms(self, text: str) -> str:
        """
        Функция вычленения ключевых слов.
        Args:
             text (str): Текст, из которого будут получены ключевые слова.
        """
        doc = self.nlp(text)
        terms = []

        matches = self.matcher(doc)
        tokens_in_phrases = set()

        for match_id, start, end in matches:
            span = doc[start:end]
            lemma_phrase = " ".join([t.lemma_ for t in span])
            terms.append(lemma_phrase)

            for i in range(start, end):
                tokens_in_phrases.add(i)

        for i, token in enumerate(doc):
            if i not in tokens_in_phrases:
                if token.pos_ in ["NOUN", "PROPN"]:
                    terms.append(token.lemma_)

        unique_terms = str(" ".join(list(set([t.capitalize() for t in terms]))))
        return unique_terms