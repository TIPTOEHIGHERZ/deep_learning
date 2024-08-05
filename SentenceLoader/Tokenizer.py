from torchtext.data.utils import get_tokenizer


class Tokenizer:
    def __init__(self, language):
        assert language in ['en_core_web_sm', 'de_core_news_sm']

        self.tokenizer = get_tokenizer('spacy', language=language)
        self.language = language
        return

    def get_tokens(self, iterator):
        if self.language == 'en_core_web_sm':
            return [self.tokenizer(sentence) for _, sentence in iterator]
        elif self.language == 'de_core_news_sm':
            return [self.tokenizer(sentence) for sentence, _ in iterator]

        return None

    def __call__(self, iterator):
        return self.get_tokens(iterator)
