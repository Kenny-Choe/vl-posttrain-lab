import copy
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from vl_posttrain.config import training_plan, validate_route
from vl_posttrain.io import read_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_package_release', ROOT / 'tools/package_release.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class PublicReleaseTests(unittest.TestCase):
    def test_plan_counts_and_independent_lineage(self):
        route = read_json(ROOT / 'configs/route.target.json')
        plan = training_plan(route)
        self.assertEqual(plan['curriculum_total_updates'], 500)
        self.assertEqual(plan['new_trajectories_per_rl_run'], 32000)
        self.assertFalse(plan['training_started'])
        invalid = copy.deepcopy(route)
        invalid['project2']['parent'] = 'P1_DPO'
        with self.assertRaises(ValueError):
            validate_route(invalid)

    def test_plan_rejects_mismatched_batch_or_update_budget(self):
        for field in ('updates', 'accumulation'):
            route = read_json(ROOT / 'configs/route.target.json')
            route['project2']['rl'][field] += 1
            with self.assertRaises(ValueError):
                validate_route(route)

    def test_local_personal_files_do_not_leak_through_packaging(self):
        base = ROOT / 'outputs/test-temp'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            root = Path(folder)
            (root / 'configs').mkdir()
            shutil.copyfile(ROOT / 'configs/publication.json', root / 'configs/publication.json')
            private_paths = ['docs/personal_notes.md', 'docs/project_showcase.html', 'configs/results.target_scenario.json',
                             'evidence/legacy_snapshot.json', 'tools/build_showcase.py', 'tests/test_artifacts.py',
                             'src/vl_posttrain/evidence/legacy.py', 'src/vl_posttrain/sample.private.py']
            for name in private_paths + ['README.md', 'src/vl_posttrain/core.py']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# example\n', encoding='utf-8')
            selected = {p.relative_to(root).as_posix() for p in packager.release_files(root)}
            self.assertTrue({'README.md', 'src/vl_posttrain/core.py'} <= selected)
            self.assertFalse(set(private_paths) & selected)

    def test_public_readme_has_route_and_code_navigation_without_result_table(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        for title in ('项目流程', '代码仓各部分', '快速开始'):
            self.assertIn(title, text)
        for personal in ('project_showcase', 'project_story', 'legacy_snapshot', 'results.target_scenario', '面试'):
            self.assertNotIn(personal, text)

    def test_published_paths_exclude_history_scores_and_private_tools(self):
        selected = {p.relative_to(ROOT).as_posix() for p in packager.release_files()}
        self.assertFalse(any(p.startswith(('evidence/', 'src/vl_posttrain/evidence/')) for p in selected))
        self.assertNotIn('tools/verify_private.py', selected)
        self.assertNotIn('configs/models.existing.json', selected)
        self.assertNotIn('docs/input_documents.json', selected)

    def test_public_ci_does_not_call_private_report_builders(self):
        workflow = (ROOT / '.github/workflows/cpu.yml').read_text(encoding='utf-8')
        self.assertNotIn('build_report.py', workflow)
        self.assertNotIn('build_showcase.py', workflow)
        self.assertIn('tools/verify.py', workflow)
