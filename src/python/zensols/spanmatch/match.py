"""Implements a method to match sections from documents to one another.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import List, Dict, Type, Tuple, Iterable, Set
from dataclasses import dataclass, field
import logging
import collections
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import ot
from scipy.spatial import distance
from zensols.nlp import FeatureToken, FeatureDocument
from zensols.datdesc import HyperparamModel
from . import DocumentMatchError, TokenPoint, MatchResult, Match

logger = logging.getLogger(__name__)


@dataclass(order=True, frozen=True)
class _BiMatch(object):
    flow: float
    forward: Match
    reverse: Match


@dataclass
class Matcher(object):
    """Creates matching spans of text between two documents by first using the
    word mover algorithm and then clustering by tokens' positions in their
    respective documents.

    """
    dtype: Type = field(default=np.float64)
    """The floating point type used for word mover and clustering."""

    hyp: HyperparamModel = field(default=None)
    """The model's hyperparameters.

    Hyperparameters::

        :param cased: whether or not to treat text as cased
        :type cased: bool

        :param distance_metric: the default distance metric for
                                calculating the distance from each
                                embedded :class:`.tokenpoint`. :see:
                                :function:`scipy.spatial.distance.cdist`
        :type distance_metric: str; one of: descendant, ancestor, all, euclidean

        :param bidirect_match: whether to order matches by a bidirectional
                               flow
        :type bidirect_match: str; one of: none, norm, sum

        :param source_distance_threshold: the source document clustering
                                          threshold distance
        :type source_distance_threshold: float

        :param target_distance_threshold: the target document clustering
                                          threshold distance
        :type target_distance_threshold: float

        :param source_position_scale: used to scale the source document
                                      positional embedding component
        :type source_position_scale: float

        :param target_position_scale: used to scale the target document
                                      positional embedding component
        :type target_position_scale: float

        :param min_flow_value: the minimum match flow; any matches that
                               fall below this value are filtered
        :type min_flow_value: float

        :param min_source_token_span: the minimum source span length in
                                      tokens to be considered for matchs
        :type min_source_token_span: int

        :param min_target_token_span: the minimum target span length in
                                      tokens to be considered for matchs
        :type min_target_token_span: int

    """
    def __post_init__(self):
        TokenPoint._CASED = self.hyp.cased

    def _nbow(self, doc: FeatureDocument) -> \
            Tuple[List[TokenPoint], Dict[str, List[TokenPoint]]]:
        """Create the nBOW (bag of words) used for document frequencies."""
        def filter_toks(t: FeatureToken) -> bool:
            return not t.is_stop and not t.is_punctuation and not t.is_space

        toks: List[TokenPoint] = []
        by_key: Dict[str, List[TokenPoint]] = collections.defaultdict(list)
        ftoks: Tuple[FeatureToken] = tuple(
            filter(filter_toks, doc.token_iter()))
        if len(ftoks) == 0:
            ftoks = doc.tokens
        for tok in ftoks:
            tp = TokenPoint(tok, doc)
            tp_key: str = tp.key
            toks.append(tp)
            by_key[tp_key].append(tp)
        return toks, dict(by_key)

    @staticmethod
    def _tok_agg(toks: List[TokenPoint], lix: int, dist: List[np.ndarray],
                 distix: List[int]):
        """Aggregate tokens' embeddings by taking the mean."""
        if toks is not None:
            emb: np.ndarray = np.concatenate(
                tuple(map(lambda t: t.embedding, toks)))
            emb: np.ndarray = np.mean(emb, axis=0)
            dist.append(emb)
            distix.append(lix)

    def _wmd(self, a: FeatureDocument, b: FeatureDocument) -> MatchResult:
        """Use the word mover algorithm to create a token to token matches.

        :param a: the source document

        :param b: the target document

        """
        aps: Tuple[List[TokenPoint]]
        bps: Tuple[List[TokenPoint]]
        atoks: Dict[str, List[TokenPoint]]
        btoks: Dict[str, List[TokenPoint]]
        aps, atoks = self._nbow(a)
        bps, btoks = self._nbow(b)
        tp_keys: List[str] = sorted(set(atoks.keys()) | set(btoks.keys()))
        n_words: int = len(tp_keys)
        hist = np.zeros((2, n_words), dtype=self.dtype)
        adist: List[np.ndarray] = []
        adistix: List[int] = []
        bdist: List[np.ndarray] = []
        bdistix: List[int] = []
        for lix, tp_key in enumerate(tp_keys):
            ats: List[TokenPoint] = atoks.get(tp_key)
            bts: List[TokenPoint] = btoks.get(tp_key)
            self._tok_agg(ats, lix, adist, adistix)
            self._tok_agg(bts, lix, bdist, bdistix)
            if ats is not None:
                hist[0, lix] = len(ats)
            if bts is not None:
                hist[1, lix] = len(bts)
        adist: np.ndarray = np.stack(adist)
        adist = adist / np.linalg.norm(adist, axis=1, keepdims=True)
        bdist: np.ndarray = np.stack(bdist)
        bdist = bdist / np.linalg.norm(bdist, axis=1, keepdims=True)
        dist_mat = distance.cdist(adist, bdist, metric=self.hyp.distance_metric)
        dist_arr: np.ndarray = np.zeros((n_words, n_words), dtype=self.dtype)
        dist_arr[np.ix_(adistix, bdistix)] = dist_mat
        if logger.isEnabledFor(logging.DEBUG):
            cnthist = hist.copy()
        hist[0] = hist[0] / hist[0].sum()
        hist[1] = hist[1] / hist[1].sum()
        if logger.isEnabledFor(logging.DEBUG):
            for i in range(hist.shape[1]):
                logger.debug(
                    f'{tp_keys[i]}: a={int(cnthist[0][i])}/{hist[0][i]}, ' +
                    f'b={int(cnthist[1][i])}/{hist[1][i]}')
        cost: np.ndarray = ot.emd(hist[0], hist[1], dist_arr)
        return MatchResult(
            source_points=aps,
            target_points=bps,
            keys=tp_keys,
            source_tokens=atoks,
            target_tokens=btoks,
            cost=cost,
            dist=dist_arr)

    def _pos_cluster(self, points: List[TokenPoint], distance_threshold: float,
                     position_scale: float) -> Dict[TokenPoint, int]:
        """Cluster a document (the source or target document) token embeddings
        using their positions.

        :param points: the token embeddings

        :return: the mapping from each point to their cluster

        """
        if len(points) == 1:
            return {points[0]: 0}
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold)
        emb: np.ndarray = np.concatenate(
            tuple(map(lambda p: p.embedding, points)))
        # normalize the embeddings to unit length
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        # add the position dimension, ascale it, and concatenate to the word
        # embeddings
        pos: np.ndarray = np.array(tuple(map(lambda p: p.position, points)))
        # scale with the hyperparameter
        pos = pos * position_scale
        emb = np.concatenate((emb, np.expand_dims(pos, axis=1)), axis=1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding shape: {emb.shape}, points: {len(points)}')
        # cluster the tokens across the embedded space and their doc position
        model.fit(emb)
        by_cluster: Dict[TokenPoint, int] = {}
        pix: int
        cid: int
        for pix, cid in enumerate(model.labels_):
            tp: TokenPoint = points[pix]
            by_cluster[tp] = cid
        return by_cluster

    def _cluster_by_position(self, res: MatchResult, fwd: bool) -> Tuple[Match]:
        """Cluster points using their posisiton in the document.

        :return: the matched document spans from the source to the target
                 document

        """
        def filter_matches(match: Match) -> bool:
            return len(match.source_tokens) >= min_src_ts and \
                len(match.target_tokens) >= min_targ_ts and \
                match.total_flow_value > h.min_flow_value

        h: HyperparamModel = self.hyp
        min_src_ts: int = h.min_source_token_span \
            if fwd else h.min_target_token_span
        min_targ_ts: int = h.min_target_token_span \
            if fwd else h.min_source_token_span
        aclusts: Dict[TokenPoint, int] = self._pos_cluster(
            res.source_points,
            h.source_distance_threshold if fwd else h.target_distance_threshold,
            h.source_position_scale if fwd else h.target_position_scale)
        bclusts: Dict[TokenPoint, int] = self._pos_cluster(
            res.target_points,
            h.target_distance_threshold if fwd else h.source_distance_threshold,
            h.target_position_scale if fwd else h.source_position_scale)
        clusts: Dict[Tuple[int, int], Match] = collections.defaultdict(Match)
        for flow in res.mapping:
            ap: TokenPoint
            for ap in flow.source_tokens:
                aclust: int = aclusts[ap]
                bp: TokenPoint
                for bp in flow.target_tokens:
                    bclust: int = bclusts[bp]
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'  {ap}({aclust}) -> {bp} {bclust}')
                    match: Match = clusts[(aclust, bclust)]
                    match.source_tokens.add(ap)
                    match.target_tokens.add(bp)
                    match.flow_values.append(flow.value)
        matches: Iterable[Match] = filter(filter_matches, clusts.values())
        return tuple(sorted(matches))

    def _reorder_bimatch(self, forward_res: MatchResult,
                         reverse_res: MatchResult) -> Tuple[Match]:
        def filter_match(m: Match) -> bool:
            if m in seen:
                return False
            seen.add(m)
            return True

        seen: Set[Match] = set()
        bd_match: str = self.hyp.bidirect_match
        bims: List[_BiMatch] = []
        forward_flows: np.ndarray = np.array(tuple(map(
            lambda m: m.total_flow_value, forward_res.matches)))
        reverse_flows: np.ndarray = np.array(tuple(map(
            lambda m: m.total_flow_value, reverse_res.matches)))
        if bd_match == 'norm':
            forward_flows = forward_flows / forward_flows.sum()
            reverse_flows = reverse_flows / reverse_flows.sum()
        elif bd_match == 'sum':
            pass
        else:
            raise DocumentMatchError(
                'Unknown bidirection match type: {bd_match}')
        for fix, fwd in enumerate(forward_res.matches):
            for rix, rev in enumerate(reverse_res.matches):
                if fwd.source_lexspan.overlaps_with(rev.target_lexspan) and \
                   fwd.target_lexspan.overlaps_with(rev.source_lexspan) or True:
                    ff: float = forward_flows[fix]
                    rf: float = reverse_flows[rix]
                    bims.append(_BiMatch(
                        flow=ff + rf,
                        forward=fwd,
                        reverse=rev))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{len(bims)} bidirectional matches found')
        if len(bims) > 0:
            bims.sort(reverse=True)
            return tuple(filter(filter_match, map(lambda bm: bm.forward, bims)))

    def match(self, source_doc: FeatureDocument,
              target_doc: FeatureDocument) -> MatchResult:
        """Match lexical spans of text from one document to the other.

        :param source_doc: the source document from where words flow

        :param target_doc: the target document to where words flow

        :return: the matched document spans from the source to the target
                 document

        """
        res: MatchResult = self._wmd(source_doc, target_doc)
        res.matches = self._cluster_by_position(res, True)
        if self.hyp.bidirect_match != 'none':
            rr_res: MatchResult = self._wmd(target_doc, source_doc)
            rr_res.matches = self._cluster_by_position(rr_res, False)
            res.matches = self._reorder_bimatch(res, rr_res)
        else:
            # sorted by lexical source by default
            res.matches = sorted(
                res.matches,
                key=lambda m: m.total_flow_value,
                reverse=True)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{len(res.matches)} matches found')
        return res

    def __call__(self, source_doc: FeatureDocument,
                 target_doc: FeatureDocument) -> MatchResult:
        """See :meth:`match`."""
        return self.match(source_doc, target_doc)
