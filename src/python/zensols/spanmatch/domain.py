"""Domain and container classes for matching document passages.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import List, Dict, ClassVar, Set, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import OrderedDict
import sys
from io import TextIOBase
import numpy as np
from torch import Tensor
from zensols.util import APIError
from zensols.config import Dictable
from zensols.persist import persisted
from zensols.nlp import FeatureToken, LexicalSpan, FeatureSpan, FeatureDocument

logger = logging.getLogger(__name__)


class DocumentMatchError(APIError):
    """Thrown for any document matching errors."""
    pass


@dataclass(eq=False)
class TokenPoint(object):
    """A token and its position in the document and in embedded space.

    """
    _CASED: ClassVar[str] = True
    """Whether to treat tokens as case sensitive."""

    token: FeatureToken = field()
    """The token used in document :obj:`doc` used for clustering."""

    doc: FeatureDocument = field()
    """The document that contains :obj:`token`."""

    def __post_init__(self):
        self._hash = hash((self.token, self.doc))

    @property
    def key(self) -> str:
        """The key used by :class:`.Matcher` used to index :class:`.WordFlow`s.

        """
        key = self.token.lemma_
        if not self._CASED:
            key = key.lower()
        return key

    @property
    @persisted('_embedding')
    def embedding(self) -> np.ndarray:
        """The token embedding."""
        if not hasattr(self.token, 'embedding'):
            raise DocumentMatchError(f'Missing embedding: {self.token}')
        tensor: Tensor = self.token.embedding
        arr: np.ndarray = tensor.cpu().detach().numpy()
        return np.expand_dims(arr.mean(axis=0), 0)

    @property
    def position(self) -> float:
        """The position of the token in the document."""
        return self.token.i / self.doc.token_len

    def __eq__(self, other: TokenPoint) -> bool:
        return self._hash == other._hash and self.token == other.token

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f'{self.token.norm}:{self.token.i}'

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(order=True)
class WordFlow(Dictable):
    """The flow of a word between two documents.

    """
    value: float = field()
    """The value of flow."""

    source_key: str = field()
    """The :obj:`.TokenPoint.key`. of the originating document."""

    target_key: str = field()
    """The :obj:`.TokenPoint.key`. of the target document."""

    source_tokens: Tuple[TokenPoint] = field(repr=False)
    """The originating tokens that map from :obj:`source_key`."""

    target_tokens: Tuple[TokenPoint] = field(repr=False)
    """The target tokens that map from :obj:`target_key`."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_tokens: bool = False):
        self._write_line(str(self), depth, writer)
        if include_tokens:
            params: Dict[str, Any] = dict(
                depth=depth + 2,
                writer=writer,
                include_type=False,
                feature_ids={'norm', 'i', 'sent_i', 'idx'},
                inline=True)
            self._write_line('source:', depth + 1, writer)
            for tok in self.source_tokens:
                tok.token.write_attributes(**params)
            self._write_line('target:', depth + 1, writer)
            for tok in self.target_tokens:
                tok.token.write_attributes(**params)

    def __str__(self) -> str:
        return f'{self.source_key} -> {self.target_key}: {self.value:.3f}'


@dataclass(order=False)
class Match(Dictable):
    """A span of matching text between two documents.

    """
    source_tokens: Set[TokenPoint] = field(default_factory=set)
    """The originating tokens from the document."""

    target_tokens: Set[TokenPoint] = field(default_factory=set)
    """The target tokens from the document"""

    flow_values: List[float] = field(default_factory=list)
    """The values of each word flow."""

    def __post_init__(self):
        self._hash = hash((tuple(sorted(self.source_tokens)),
                           tuple(sorted(self.target_tokens)),
                           tuple(self.flow_values)))

    @property
    def total_flow_value(self) -> float:
        """The sum of the :obj:`flow_values`."""
        return sum(self.flow_values)

    @property
    def source_document(self) -> FeatureDocument:
        """The originating document."""
        return next(iter(self.source_tokens)).doc

    @property
    def target_document(self) -> FeatureDocument:
        """The target document."""
        return next(iter(self.target_tokens)).doc

    def _get_lexspan(self, tokens: Set[TokenPoint], doc: FeatureDocument) -> \
            FeatureSpan:
        return LexicalSpan.widen(map(lambda t: t.token.lexspan, tokens))

    @property
    def source_lexspan(self) -> LexicalSpan:
        """The originating document's lexical span."""
        return self._get_lexspan(self.source_tokens, self.source_document)

    @property
    def target_lexspan(self) -> LexicalSpan:
        """The target document's lexical span."""
        return self._get_lexspan(self.target_tokens, self.target_document)

    @property
    def source_span(self) -> FeatureSpan:
        """The originating document's span."""
        return self.source_document.get_overlapping_span(
            self.source_lexspan, inclusive=False).to_sentence()

    @property
    def target_span(self) -> FeatureSpan:
        """The target document's span."""
        return self.target_document.get_overlapping_span(
            self.target_lexspan, inclusive=False).to_sentence()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_tokens: bool = False,
              include_flow: bool = True,
              char_limit: int = sys.maxsize):
        if include_flow:
            self._write_line(f'flow: {self.total_flow_value}', depth, writer)
        if include_tokens:
            self._write_line('source:', depth, writer)
            for tok in self.source_tokens:
                self._write_line(tok, depth + 1, writer)
            self._write_line('target:', depth, writer)
            for tok in self.target_tokens:
                self._write_line(tok, depth + 1, writer)
        else:
            self._write_line(f'source {self.source_lexspan}:',
                             depth, writer)
            self.source_span.write_text(depth + 1, writer, limit=char_limit)
            self._write_line(f'target {self.target_lexspan}:',
                             depth, writer)
            self.target_span.write_text(depth + 1, writer, limit=char_limit)

    def asflatdict(self, *args,
                   include_norm: bool = False,
                   include_text: bool = False,
                   **kwargs):
        dct = OrderedDict(
            [['flow', float(self.total_flow_value)],
             ['source_span', self.source_lexspan.asflatdict()],
             ['target_span', self.target_lexspan.asflatdict()]])
        if include_norm:
            dct['source_norm'] = self.source_span.norm
            dct['target_norm'] = self.target_span.norm
        if include_text:
            dct['source_text'] = self.source_span.text
            dct['target_text'] = self.target_span.text
        return dct

    def to_str(self, tokens: bool = False, spans: bool = True,
               flow: bool = True) -> str:
        s: str
        if tokens:
            s = f'{self.source_tokens} -> {self.target_tokens}'
        else:
            if spans:
                s = (f'{self.source_span}{self.source_lexspan} -> ' +
                     f'{self.target_span}{self.target_lexspan}')
            else:
                s = f'{self.source_span} -> {self.target_span}'
        if flow:
            s = s + f', flow={self.total_flow_value:.3e}'
        return s

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Match) -> bool:
        return self.flow_values == other.flow_values and \
            self.source_tokens == other.source_tokens and \
            self.target_tokens == other.target_tokens

    def __lt__(self, other: Match) -> bool:
        return self.source_lexspan < other.source_lexspan

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class MatchResult(Dictable):
    """Contains the lexical text match pairs from the first to the second
    document given by :meth:`.Matcher.match`.

    """
    _DICTABLE_WRITE_EXCLUDES: ClassVar[Set[str]] = set(
        'keys cost dist'.split())
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = set('flows'.split())

    keys: List[str] = field(repr=False)
    """The :obj:`.TokenPoint.key`s to tokens used to normalize document
    frequencies in the nBOW.

    """
    source_points: List[TokenPoint] = field(repr=False)
    """The first document's token points."""

    target_points: List[TokenPoint] = field(repr=False)
    """The second  document's token points."""

    source_tokens: Dict[str, List[TokenPoint]] = field(repr=False)
    """The first document's token points indexed by the :obj:`.TokenPoint.key`.

    """
    target_tokens: Dict[str, List[TokenPoint]] = field(repr=False)
    """The first document's token points indexed by the :obj:`.TokenPoint.key`.

    """
    cost: np.ndarray = field(repr=False)
    """The earth mover distance solution, which is the cost of transportation
    from first to the second document.

    """
    dist: np.ndarray = field(repr=False)
    """The distance matrix of all token's in the embedded space."""

    matches: Tuple[Match] = field(default=None)
    """The matching passages between the documents."""

    @property
    @persisted('_transit', transient=True)
    def transit(self) -> np.ndarray:
        return self.cost * self.dist

    @property
    @persisted('_flows', transient=True)
    def flows(self) -> Tuple[WordFlow]:
        """The Word Mover positional flows."""
        trans: np.ndarray = self.transit
        paths: np.ndarray = np.nonzero(trans)
        wflows: List[WordFlow] = []
        for r, c in zip(paths[0], paths[1]):
            fr: str = self.keys[r]
            to: str = self.keys[c]
            wflows.append(WordFlow(
                source_key=fr,
                target_key=to,
                source_tokens=self.source_tokens[fr],
                target_tokens=self.target_tokens[to],
                value=trans[r, c]))
        wflows.sort(reverse=True)
        return tuple(wflows)

    @property
    @persisted('_mapping', transient=True)
    def mapping(self) -> Tuple[WordFlow]:
        """Like :obj:`flows` but do not duplicate sources"""
        srcs: Set[str] = set()
        flows: List[WordFlow] = []
        flow: WordFlow
        for flow in self.flows:
            if flow.source_key not in srcs:
                flows.append(flow)
                srcs.add(flow.source_key)
        return tuple(flows)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_source: bool = True, include_target: bool = True,
              include_tokens: bool = False, include_mapping: bool = True,
              match_detail: bool = False):
        flow_val: float = sum(map(lambda f: f.value, self.flows))
        mapping: float = sum(map(lambda f: f.value, self.mapping))
        if include_source:
            self._write_line('source:', depth, writer)
            self.source_points[0].doc.write_text(depth + 1, writer)
        if include_target:
            self._write_line('target:', depth, writer)
            self.target_points[0].doc.write_text(depth + 1, writer)
        self._write_line('flow:', depth, writer)
        self._write_line(f'mapped: {mapping}', depth + 1, writer)
        self._write_line(f'total: {flow_val}', depth + 1, writer)
        if include_mapping:
            self._write_line('mapping:', depth, writer)
            flow: WordFlow
            for flow in self.mapping:
                flow.write(depth + 1, writer, include_tokens=include_tokens)
        self._write_line('matches:', depth, writer)
        match: Match
        for i, match in enumerate(self.matches):
            if match_detail:
                self._write_line(f'{i}:', depth + 1, writer)
                match.write(depth + 2, writer)
            else:
                self._write_line(match, depth + 1, writer)
