from typing import Dict, Any, Tuple
import json
import unittest
from pathlib import Path
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.spanmatch import Match, MatchResult, Matcher


class TestMatch(unittest.TestCase):
    def setUp(self):
        conf = ImportIniConfig('test-resources/spanmatch.conf')
        self.fac = ImportConfigFactory(conf)
        self.maxDiff = 999999

    def _get_article(self) -> Dict[str, Any]:
        with open('test-resources/article.json') as f:
            return json.load(f)

    def test_match(self):
        save: bool = False
        should_file: Path = Path('test-resources/match.json')
        content: Dict[str, Any] = self._get_article()
        doc_parser: FeatureDocumentParser = self.fac('spanmatch_doc_parser')
        matcher: Matcher = self.fac('spanmatch_matcher')
        source: FeatureDocument = doc_parser(content['source'])
        summary: FeatureDocument = doc_parser(content['summary'])
        res: MatchResult = matcher.match(source, summary)
        matches: Tuple[Match] = res.matches
        match: Match = matches[1]
        test_tups = [list(match.source_span.norm_token_iter()),
                     list(match.target_span.norm_token_iter())]
        if save:
            from pprint import pprint
            pprint(test_tups)
            should_tups = test_tups
            with open(should_file, 'w') as f:
                json.dump(should_tups, f, indent=4)
        else:
            with open(should_file) as f:
                should_tups = json.load(f)
        self.assertEqual(should_tups, test_tups)
