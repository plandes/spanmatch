#!/usr/bin/env python

from zensols.cli import CliHarness
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.spanmatch import Match, MatchResult, Matcher, ApplicationFactory


SOURCE = """\
Johannes Gutenberg (1398 – 1468) was a German goldsmith and publisher who
introduced printing to Europe. His introduction of mechanical movable type
printing to Europe started the Printing Revolution and is widely regarded as the
most important event of the modern period. It played a key role in the
scientific revolution and laid the basis for the modern knowledge-based economy
and the spread of learning to the masses.

Gutenberg many contributions to printing are: the invention of a process for
mass-producing movable type, the use of oil-based ink for printing books,
adjustable molds, and the use of a wooden printing press. His truly epochal
invention was the combination of these elements into a practical system that
allowed the mass production of printed books and was economically viable for
printers and readers alike.
"""

SUMMARY = """\
The German Johannes Gutenberg introduced printing in Europe. His invention had a
decisive contribution in spread of mass-learning and in building the basis of
the modern society.
"""


def main():
    harness: CliHarness = ApplicationFactory.create_harness()
    doc_parser: FeatureDocumentParser = harness['spanmatch_doc_parser']
    matcher: Matcher = harness['spanmatch_matcher']
    source: FeatureDocument = doc_parser(SOURCE)
    summary: FeatureDocument = doc_parser(SUMMARY)
    # shorten source doc span length by scaling up positional importance
    matcher.hyp.source_position_scale = 2.5
    # elongate summary doc span length by scaling up positional importance
    matcher.hyp.target_position_scale = 0.9
    res: MatchResult = matcher(source, summary)
    match: Match
    for i, match in enumerate(res.matches[:5]):
        if i > 0:
            print('_' * 40)
        match.write(include_flow=False)


if (__name__ == '__main__'):
    main()
