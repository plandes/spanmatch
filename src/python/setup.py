from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.spanmatch",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    package_data={'': ['*.conf', '*.json', '*.yml']},
    description='An API to match spans of semantically similar text across documents.',
    user='plandes',
    project='spanmatch',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
