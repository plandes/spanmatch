"""An API to match spans of semantically similar text across documents.

"""
__author__ = 'Paul Landes'

from typing import Callable, List, Any, Dict, Tuple, ClassVar
from dataclasses import dataclass, field
from enum import Enum, auto
import sys
import logging
from pathlib import Path
import json
import yaml
from zensols.util import stdout
from zensols.cli import ApplicationError
from zensols.introspect import IntegerSelection
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.datdesc import HyperparamModel
from . import Match, MatchResult, Matcher

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    text = auto()
    sphinx = auto()
    json = auto()
    yaml = auto()


@dataclass
class Application(object):
    """An API to match spans of semantically similar text across documents.

    """
    doc_parser: FeatureDocumentParser = field()
    """The feature document that normalizes (whitespace) parsed documents."""

    matcher: Matcher = field()
    """Used to match spans of text."""

    def _read_and_parser_file(self, path: Path) -> FeatureDocument:
        with open(path) as f:
            content: str = f.read()
        return self.doc_parser(content)

    def match(self, source_file: Path, target_file: Path,
              output_format: OutputFormat = OutputFormat.text,
              selection: IntegerSelection = IntegerSelection('0'),
              output_file: Path = Path('-'),
              detail: bool = False):
        """Match spans across two text files.

        :param source_file: the source match file

        :param target_file: the target match file

        :param output_format: the format to write the hyperparemters

        :param selection: the matches to output

        :param output_file: the output file or ``-`` for standard out

        :param detail: whether to output more information

        """
        source: FeatureDocument = self._read_and_parser_file(source_file)
        target: FeatureDocument = self._read_and_parser_file(target_file)
        res: MatchResult = self.matcher(source, target)
        matches: List[Match] = selection(res.matches)
        line: str = (('_' * 79) + '\n')
        with stdout(output_file) as f:
            if output_format == OutputFormat.text:
                match: Match
                for i, match in enumerate(matches):
                    if detail:
                        f.write(f'<{match.source_span.norm}> -> ' +
                                f'<{match.target_span.norm}>\n')
                        match.write(depth=1, writer=f)
                    else:
                        if i > 0:
                            f.write(line)
                        f.write(f'source: <{match.source_span.norm}>\n')
                        f.write(f'target: <{match.target_span.norm}>\n')
            elif output_format in {OutputFormat.json, OutputFormat.yaml}:
                mdcts: Tuple[Dict[str, Any]] = list(
                    map(lambda m: m.asflatdict(include_norm=detail), matches))
                if OutputFormat.json == output_format:
                    json.dump({'matches': mdcts}, f, indent=4)
                else:
                    yaml.dump({'matches': list(map(dict, mdcts))}, f)
            else:
                raise ApplicationError(
                    f'Unsupported format: {output_format.name}')

    def write_hyperparam(self, output_format: OutputFormat = OutputFormat.text):
        """Write the matcher's hyperparameter documentation.

        :param output_format: the format to write the hyperparemters

        """
        hyp: HyperparamModel = self.matcher.hyp
        fn: Callable = {
            OutputFormat.text: lambda: hyp.write(include_doc=True),
            OutputFormat.sphinx: lambda: hyp.write_sphinx(),
            OutputFormat.json: lambda: hyp.asjson(writer=sys.stdout, indent=4),
            OutputFormat.yaml: lambda: hyp.asyaml(writer=sys.stdout),
        }[output_format]
        fn()


@dataclass
class ProtoApplication(object):
    CLI_META: ClassVar[Dict[str, Any]] = {'is_usage_visible': False}

    app: Application = field()

    def _match(self):
        self.app.match(
            Path('test-resources/source.txt'),
            Path('test-resources/summary.txt'),
            output_format=OutputFormat.text,
            selection=IntegerSelection('2:4'))

    def proto(self):
        """Used for REPL prototyping."""
        self._match()
