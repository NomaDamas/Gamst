import sys
import types
import unittest
from unittest.mock import patch

from gamst import cli


class CliLocalCommands(unittest.TestCase):
    def test_install_is_an_explicit_dispatch(self):
        with patch.object(cli, "_dispatch_local_command") as dispatch, patch.object(
            sys, "argv", ["gamst", "install", "status"]
        ):
            cli.main()
        dispatch.assert_called_once_with("install", ["status"])

    def test_run_is_an_explicit_dispatch_and_forwards_flags(self):
        with patch.object(cli, "_dispatch_local_command") as dispatch, patch.object(
            sys, "argv", ["gamst", "run", "--language", "ko"]
        ):
            cli.main()
        dispatch.assert_called_once_with("run", ["--language", "ko"])

    def test_dispatch_uses_checkout_python_and_propagates_child_failure(self):
        script = cli._LOCAL_COMMANDS["run"]
        completed = types.SimpleNamespace(returncode=17)
        with patch("gamst.cli.subprocess.run", return_value=completed) as run:
            with self.assertRaises(SystemExit) as raised:
                cli._dispatch_local_command("run", ["--stop"])
        self.assertEqual(raised.exception.code, 17)
        run.assert_called_once_with(
            [sys.executable, str(script), "--stop"],
            cwd=str(cli._CHECKOUT_ROOT),
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
