# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import logging
import random
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5EncoderModel, T5Tokenizer


from .base import _BaseTextConditioner, ConditionType, TokenizedText, EmbeddedText


logger = logging.getLogger(__name__)


def length_to_mask(lengths: torch.Tensor, max_len: tp.Optional[int] = None) -> torch.Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]


def hash_trick(word: str, vocab_size: int) -> int:
    """Hash trick to pair each word with an index

    Args:
        word (str): word we wish to convert to an index
        vocab_size (int): size of the vocabulary
    Returns:
        int: index of the word in the embedding LUT
    """
    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size


class TorchAutocast:
    """TorchAutocast utility class.
    Allows you to enable and disable autocast. This is specially useful
    when dealing with different architectures and clusters with different
    levels of support.

    Args:
        enabled (bool): Whether to enable torch.autocast or not.
        args: Additional args for torch.autocast.
        kwargs: Additional kwargs for torch.autocast
    """

    def __init__(self, enabled: bool, *args, **kwargs):
        self.autocast = torch.autocast(*args, **kwargs) if enabled else None

    def __enter__(self):
        if self.autocast is None:
            return
        try:
            self.autocast.__enter__()
        except RuntimeError:
            device = self.autocast.device
            dtype = self.autocast.fast_dtype
            raise RuntimeError(
                f"There was an error autocasting with dtype={dtype} device={device}\n"
                "If you are on the FAIR Cluster, you might need to use autocast_dtype=float16"
            )

    def __exit__(self, *args, **kwargs):
        if self.autocast is None:
            return
        self.autocast.__exit__(*args, **kwargs)


class TextConditioner(_BaseTextConditioner[TokenizedText]): ...


class Tokenizer:
    """Base tokenizer implementation"""

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]

    Args:
        n_bins (int): we use a hash modulo the number of bins of the tokens. Should set
            the number of bins to use.
        language (str or None): if provided, name of a spacy model for doing text normalization.
            Must be provided if `lemma` or `stopwords` is True.
        lemma (bool): whether to lemmatize the text.
        stopwords (bool): whether to remove stopwords.

    """

    PUNCTUATION = "?:!.,;"

    def __init__(
        self,
        n_bins: int,
        language: tp.Optional[str] = "en_core_web_sm",
        lemma: bool = True,
        stopwords: bool = True,
    ) -> None:
        # Lazy import because spacy versions are really annoying and incompatible with Mistral API package.
        import spacy

        self.n_bins = n_bins
        self.lemma = lemma
        self.stopwords = stopwords
        self.pad_idx = n_bins
        if language is None:
            assert not lemma and not stopwords
            self.language_model = None
        else:
            try:
                self.language_model = spacy.load(language)
            except IOError:
                spacy.cli.download(language)  # type: ignore
                self.language_model = spacy.load(language)

    def process_text(self, text: str) -> tp.List[str]:
        if self.language_model is None:
            return text.split()
        else:
            # normalize text
            words = self.language_model(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                words = [word for word in words if not word.is_stop]  # type: ignore
            # remove punctuation
            words = [word for word in words if word.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            if self.lemma:
                out_text = [word.lemma_ for word in words]
            else:
                out_text = [word.text for word in words]

            return out_text

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are.
        """
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            words = self.process_text(text)
            lengths.append(len(words))
            # convert to tensor
            tokens = torch.tensor([hash_trick(word, self.n_bins) for word in words])
            output.append(tokens)

        mask = length_to_mask(torch.tensor(lengths))
        padded_output = pad_sequence(output, padding_value=self.pad_idx, batch_first=True).int()
        return TokenizedText(padded_output, mask)


class NoopTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]

    When all possible values are known, one can use `possible_values` to provide the list
    of possible tokens. If a token doesn't exist, `pad_idx` will be used instead.
    """

    def __init__(self, n_bins: int, possible_values: list[str] | None = None):
        self.n_bins = n_bins
        self.pad_idx = n_bins
        if possible_values is None:
            self.possible_values = None
        else:
            self.possible_values = {value: idx for idx, value in enumerate(possible_values)}
            assert n_bins >= len(possible_values)

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                if self.possible_values is None:
                    output.append(hash_trick(text, self.n_bins))
                else:
                    if text not in self.possible_values:
                        raise ValueError(f"'{text}' is not in possible_values {self.possible_values}")
                    output.append(self.possible_values[text])
                lengths.append(1)

        tokens = torch.tensor(output).int()[:, None]
        mask = length_to_mask(torch.tensor(lengths))
        return TokenizedText(tokens, mask)


class LUTConditioner(TextConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """

    def __init__(
        self, n_bins: int, tokenizer: str, possible_values: list[str] | None = None, init_scale: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(n_bins + 1, self.dim)  # n_bins + 1 for padding.
        self.embed.weight.data *= init_scale
        if tokenizer == "noop":
            self.tokenizer = NoopTokenizer(n_bins, possible_values)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return TokenizedText(tokens.to(device), mask.to(device))

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        return ConditionType(embeds, mask)


class T5Conditioner(TextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """

    MODELS = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    ENCODER_CLS = T5EncoderModel
    TOKENIZER_CLS = T5Tokenizer

    def __init__(
        self,
        name: str,
        finetune: bool = False,
        autocast_dtype: tp.Optional[str] = "float32",
        word_dropout: float = 0.0,
        normalize_text: bool = False,
        **kwargs,
    ):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(dim=self.MODELS_DIMS[name], **kwargs)
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        if autocast_dtype is None or self.device == "cpu":
            self.autocast = TorchAutocast(enabled=False)
            if self.device != "cpu":
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = self.TOKENIZER_CLS.from_pretrained(name)
                t5 = self.ENCODER_CLS.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)
        if finetune:
            self.t5 = t5
        else:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            self.__dict__["t5"] = t5.to(self.device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def prepare(self, x: tp.List[tp.Optional[str]], *args: tp.Any, **kwargs: tp.Any) -> EmbeddedText:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            entries = [" ".join(self.text_normalizer.process_text(entry)) for entry in entries]
        if self.word_dropout > 0.0 and self.training:
            new_entries = []
            for entry in entries:
                if self.word_dropout:
                    words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                    entry = " ".join(words)
                new_entries.append(entry)
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True).to(self.device)
        mask = inputs["attention_mask"]
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant

        tokens = inputs["input_ids"]
        with torch.set_grad_enabled(False), self.autocast:
            t5_embeds = self.t5.shared(tokens)
        return EmbeddedText(t5_embeds, mask)

    def _get_condition(self, tokenized: EmbeddedText) -> ConditionType:
        embeds, mask = tokenized
        inputs = {
            "attention_mask": mask,
            "inputs_embeds": embeds,
        }
        mask = inputs["attention_mask"]
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        return ConditionType(embeds, mask)


class MultiT5Conditioner(T5Conditioner):
    """T5-based conditioner that handles multiple conditioning texts for one sample. This conditioner takes a list of reference texts.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
        ref_dropout (float, optional): Reference dropout probability. Not useful at inference time.
        frame_rate (float, optional): Frame rate for converting time to frame indices. Defaults to 12.5. Not useful at inference time.
        rag_time_sampling_params (dict, optional): Parameters for RAG time sampling. Not useful at inference time.
    """

    def __init__(
        self,
        name: str,
        finetune: bool = False,
        autocast_dtype: tp.Optional[str] = "float32",
        word_dropout: float = 0.0,
        normalize_text: bool = False,
        ref_dropout: float = 0.0,
        frame_rate: float = 12.5,
        rag_time_sampling_params: tp.Optional[tp.Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            finetune=finetune,
            autocast_dtype=autocast_dtype,
            word_dropout=word_dropout,
            normalize_text=normalize_text,
            **kwargs,
        )

    def _get_condition(
        self,
        inputs: EmbeddedText,
    ) -> ConditionType:
        """Get condition tensor from multiple references.

        Args:
            inputs: EmbeddedText

        Returns:
            ConditionType containing condition and mask
        """
        embeddings, raw_mask = super()._get_condition(inputs)
        B = inputs.embeddings.shape[0]
        assert B == 1, "MultiT5Conditioner only supports one reference for now"

        # Check whether the reference is empty with input_mask since output_mask may still has non-zero values even if the input is an empty string
        if inputs.mask.sum() != 0:
            # raw_mask is all True if inputs is not empty
            embeddings = embeddings * raw_mask.unsqueeze(2)
            return ConditionType(embeddings, raw_mask)
        else:
            # Set mask to False when the reference is empty since this is where we would like to apply the learnt padding if self.learn_padding is True
            return ConditionType(torch.zeros_like(embeddings), torch.zeros_like(raw_mask))
