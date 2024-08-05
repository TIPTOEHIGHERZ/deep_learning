import torch
import torchtext
from .Tokenizer import Tokenizer
import itertools


def collate_fn(data: list):
    x = list()
    y = list()
    for x_, y_ in data:
        x.append(x_)
        y.append(y_)

    return torch.tensor(x), torch.tensor(y)


class SentenceLoader:
    tokens_en: list = None
    tokens_ge: list = None
    vocab_en: torchtext.vocab.Vocab = None
    vocab_ge: torchtext.vocab.Vocab = None
    numeral_token_en: list = None
    numeral_token_ge: list = None

    def __init__(self, data_iterator, shuffle=False, batch_size=1, device='cuda'):
        self.tokenizer_en = Tokenizer('en_core_web_sm')
        self.tokenizer_ge = Tokenizer('de_core_news_sm')
        self.data = list(data_iterator)

        self.make_tokens()
        self.build_vocab()

        self.numeral_token_en = self.text_transform(self.tokens_en, 'en')
        self.numeral_token_en = self.text_pad(self.numeral_token_en, 512, 'en')
        self.numeral_token_ge = self.text_transform(self.tokens_ge, 'ge')
        self.numeral_token_ge = self.text_pad(self.numeral_token_ge, 512, 'ge')

        self.data_loader = torch.utils.data.DataLoader(list(zip(self.numeral_token_en, self.numeral_token_ge)),
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       collate_fn=collate_fn)

        return

    def make_tokens(self):
        self.tokens_en = self.tokenizer_en(self.data)
        self.tokens_ge = self.tokenizer_ge(self.data)

        return self.tokens_en, self.tokens_ge

    def build_vocab(self, specials=['<unk>', '<BOS>', '<EOS>', '<PAD>']):
        self.vocab_en = torchtext.vocab.build_vocab_from_iterator(self.tokens_en, specials=specials)
        self.vocab_en.set_default_index(self.vocab_en['<unk>'])
        self.vocab_ge = torchtext.vocab.build_vocab_from_iterator(self.tokens_ge, specials=specials)
        self.vocab_ge.set_default_index(self.vocab_ge['<unk>'])

        return

    def text_transform(self, sentence_list: list, language: str):
        if language == 'en':
            return [[self.vocab_en['<BOS>']] + [self.vocab_en[token] for token in tokens] + [self.vocab_en['<EOS>']]
                    for tokens in sentence_list]
        elif language == 'ge':
            return [[self.vocab_ge['<BOS>']] + [self.vocab_ge[token] for token in tokens] + [self.vocab_ge['<EOS>']]
                    for tokens in sentence_list]

        return None

    def text_pad(self, tokens: list, fixed_length: int, language: str):
        if language == 'en':
            vocab = self.vocab_en
        elif language == 'ge':
            vocab = self.vocab_ge
        else:
            return None

        for token in tokens:
            if fixed_length < len(token):
                print('fixed_length too small')
                raise ValueError
            token += (fixed_length - len(token)) * [vocab['<PAD>']]

        return tokens

    def __iter__(self):
        return iter(self.data_loader)

    def get_batch(self, index):
        return next(itertools.islice(iter(self.data_loader), index, index + 1))
