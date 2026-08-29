#![cfg(unix)]

use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::Value;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    fake_ssh: PathBuf,
    argv_log: PathBuf,
    stdin_log: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("rdbg-integration-{}-{id}", std::process::id()));
        fs::create_dir_all(&root).unwrap();

        // The executable path intentionally contains a space. `rdbg` must pass it
        // directly to Command::new rather than interpolate it into a shell command.
        let fake_ssh = root.join("fake ssh");
        let argv_log = root.join("argv.log");
        let stdin_log = root.join("stdin.log");
        fs::write(
            &fake_ssh,
            r#"#!/bin/sh
set -eu
: > "$RDBG_FAKE_ARGV"
for arg in "$@"; do
    printf '%s\n' "$arg" >> "$RDBG_FAKE_ARGV"
done
cat > "$RDBG_FAKE_STDIN"
printf '%s' "${RDBG_FAKE_STDOUT-}"
printf '%s' "${RDBG_FAKE_STDERR-}" >&2
if [ -n "${RDBG_FAKE_STDOUT_BYTES-}" ]; then
    dd if=/dev/zero bs="$RDBG_FAKE_STDOUT_BYTES" count=1 2>/dev/null | tr '\000' x
fi
if [ -n "${RDBG_FAKE_STDERR_BYTES-}" ]; then
    dd if=/dev/zero bs="$RDBG_FAKE_STDERR_BYTES" count=1 2>/dev/null | tr '\000' y >&2
fi
if [ -n "${RDBG_FAKE_DESCENDANT_SLEEP-}" ]; then
    (sleep "$RDBG_FAKE_DESCENDANT_SLEEP") &
    exit "${RDBG_FAKE_EXIT-0}"
fi
if [ -n "${RDBG_FAKE_SLEEP-}" ]; then
    exec sleep "$RDBG_FAKE_SLEEP"
fi
exit "${RDBG_FAKE_EXIT-0}"
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_ssh).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_ssh, permissions).unwrap();

        // The first execution of a freshly written script can stall well past
        // a second (macOS assesses new executables on first exec; loaded CI
        // runners add scheduling delay). Pay that cost here so tests with a
        // deadline never race the stall.
        // Warmup logs stay separate: some tests assert the real argv log was
        // never created as proof that ssh was not launched.
        let warmup = Self::warmup(&fake_ssh, &root);
        assert!(warmup.success(), "fake ssh warmup failed: {warmup:?}");

        Self {
            root,
            fake_ssh,
            argv_log,
            stdin_log,
        }
    }

    /// Execute the freshly written script once.
    ///
    /// Cargo runs integration tests on parallel threads, so another test can fork
    /// while this thread still holds the script open for writing; the child inherits
    /// that write descriptor and exec fails with `ETXTBSY` until it exits. The window
    /// is short, so retry within a deadline instead of failing the run.
    fn warmup(fake_ssh: &Path, root: &Path) -> ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let launched = Command::new(fake_ssh)
                .env("RDBG_FAKE_ARGV", root.join("warmup-argv.log"))
                .env("RDBG_FAKE_STDIN", root.join("warmup-stdin.log"))
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            match launched {
                Ok(status) => return status,
                Err(error)
                    if error.kind() == ErrorKind::ExecutableFileBusy
                        && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(error) => panic!("fake ssh warmup could not launch: {error}"),
            }
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_rdbg"));
        command
            .arg("--ssh")
            .arg(&self.fake_ssh)
            .env("RDBG_FAKE_ARGV", &self.argv_log)
            .env("RDBG_FAKE_STDIN", &self.stdin_log)
            .env_remove("RDBG_FAKE_STDOUT")
            .env_remove("RDBG_FAKE_STDERR")
            .env_remove("RDBG_FAKE_STDOUT_BYTES")
            .env_remove("RDBG_FAKE_STDERR_BYTES")
            .env_remove("RDBG_FAKE_DESCENDANT_SLEEP")
            .env_remove("RDBG_FAKE_SLEEP")
            .env_remove("RDBG_FAKE_EXIT");
        command
    }

    fn argv(&self) -> Vec<String> {
        fs::read_to_string(&self.argv_log)
            .unwrap()
            .lines()
            .map(ToOwned::to_owned)
            .collect()
    }

    fn payload(&self) -> Vec<u8> {
        fs::read(&self.stdin_log).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn assert_static_ssh_argv(argv: &[String], target: &str) {
    assert_eq!(argv.len(), 18, "argv: {argv:#?}");
    assert_eq!(
        &argv[..17],
        [
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            target,
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
        ]
    );
    assert_eq!(
        decode_utf16_base64(argv[17].as_bytes()),
        rdbg::POWERSHELL_BOOTSTRAP
    );
}

fn assert_bare_ssh_argv(argv: &[String], target: &str) {
    assert_eq!(argv.len(), 8, "argv: {argv:#?}");
    assert_eq!(
        &argv[..7],
        [
            "-T",
            target,
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
        ]
    );
    assert_eq!(
        decode_utf16_base64(argv[7].as_bytes()),
        rdbg::POWERSHELL_BOOTSTRAP
    );
}

fn decode_utf16_base64(encoded: &[u8]) -> String {
    assert!(encoded.is_ascii());
    let bytes = BASE64.decode(encoded).unwrap();
    assert_eq!(bytes.len() % 2, 0);
    let units = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    String::from_utf16(&units).unwrap()
}

fn parse_json(output: &Output) -> Value {
    assert!(
        output.stderr.is_empty(),
        "local stderr: {:?}",
        output.stderr
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["schema"], "rdbg.result.v1");
    result
}

fn run_with_stdin(mut command: Command, input: &[u8]) -> Output {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn eval_preserves_unicode_variables_quotes_backticks_json_and_multiline() {
    let fixture = Fixture::new();
    let script = r#"$名字 = "O'Reilly"
$single = 'C:\Program Files\工具\one.ps1'
$json = '{"kind":"test","value":"$notExpanded","quote":"\""}'
Write-Output "`"中文`" / `$literal / $名字"
Write-Output `here
"#;

    let output = fixture
        .command()
        .args(["ps", "Z3-alias", "--eval", script])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_OK));
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    let argv = fixture.argv();
    assert_static_ssh_argv(&argv, "Z3-alias");
    assert!(!argv.iter().any(|arg| arg.contains("O'Reilly")));
    assert_eq!(decode_utf16_base64(&fixture.payload()), script);
}

#[test]
fn file_with_spaces_uses_the_same_static_argv_and_payload() {
    let fixture = Fixture::new();
    let script_dir = fixture.root.join("scripts with spaces");
    fs::create_dir(&script_dir).unwrap();
    let script_path = script_dir.join("unicode check.ps1");
    let script = r#"$x = '中文 and ''single quotes'''\nWrite-Output "$x `"quoted`""\n"#;
    fs::write(&script_path, script).unwrap();

    let output = fixture
        .command()
        .arg("ps")
        .arg("zzz")
        .arg("--file")
        .arg(&script_path)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_OK));
    assert_static_ssh_argv(&fixture.argv(), "zzz");
    assert_eq!(decode_utf16_base64(&fixture.payload()), script);
}

#[test]
fn large_multiline_file_bypasses_command_line_length_limits() {
    let fixture = Fixture::new();
    let line = r#"$value = 'LONG_PAYLOAD_SENTINEL 中文 $single `"double`" {"json":true}'
"#;
    let script = line.repeat(2_000);
    assert!(
        script.len() > 100 * 1024,
        "script was only {} bytes",
        script.len()
    );
    let script_path = fixture.root.join("large multiline script.ps1");
    fs::write(&script_path, &script).unwrap();

    let output = fixture
        .command()
        .arg("ps")
        .arg("Z3")
        .arg("--file")
        .arg(&script_path)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_OK));
    let argv = fixture.argv();
    assert_static_ssh_argv(&argv, "Z3");
    assert!(
        argv.iter()
            .all(|argument| !argument.contains("LONG_PAYLOAD_SENTINEL"))
    );
    assert_eq!(decode_utf16_base64(&fixture.payload()), script);
}

#[test]
fn file_dash_reads_local_stdin_then_sends_ascii_payload_to_ssh() {
    let fixture = Fixture::new();
    let script = "Write-Output 'stdin 中文'\n$object = @{ json = '{\"a\":1}' }\n";
    let mut command = fixture.command();
    command.args(["ps", "Z3", "--file", "-"]);

    let output = run_with_stdin(command, script.as_bytes());

    assert_eq!(output.status.code(), Some(rdbg::EXIT_OK));
    assert_static_ssh_argv(&fixture.argv(), "Z3");
    let wire_payload = fixture.payload();
    assert!(wire_payload.is_ascii());
    assert_eq!(decode_utf16_base64(&wire_payload), script);
}

#[test]
fn json_captures_remote_stdout_stderr_and_real_status() {
    let fixture = Fixture::new();
    let remote_stdout = "{\"remote\":\"中文\",\"quoted\":\"\\\"\"}\n";
    let remote_stderr = "PowerShell error: '$x' and `tick`\n";
    let output = fixture
        .command()
        .env("RDBG_FAKE_STDOUT", remote_stdout)
        .env("RDBG_FAKE_STDERR", remote_stderr)
        .env("RDBG_FAKE_EXIT", "7")
        .args(["--json", "ps", "Z3", "--eval", "exit 7"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_REMOTE_COMMAND));
    let result = parse_json(&output);
    assert_eq!(result["ok"], false);
    assert_eq!(result["error_class"], "remote_command");
    assert_eq!(result["rdbg_exit_code"], rdbg::EXIT_REMOTE_COMMAND);
    assert_eq!(result["process_exit_code"], 7);
    assert_eq!(result["stdout"], remote_stdout);
    assert_eq!(result["stderr"], remote_stderr);
    assert_eq!(result["stdout_utf8_lossy"], false);
    assert_eq!(result["stderr_utf8_lossy"], false);
}

#[test]
fn human_mode_replays_captured_output() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .env("RDBG_FAKE_STDOUT", "stdout 中文\n")
        .env("RDBG_FAKE_STDERR", "stderr warning\n")
        .args(["ps", "Z3", "--eval", "Write-Output ok"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_OK));
    assert_eq!(output.stdout, b"stdout \xe4\xb8\xad\xe6\x96\x87\n");
    assert_eq!(output.stderr, b"stderr warning\n");
}

#[test]
fn timeout_terminates_ssh_and_retains_captured_output() {
    let fixture = Fixture::new();
    let remote_stdout = "partial stdout before sleep 中文\n";
    let remote_stderr = "partial stderr before sleep\n";
    let started_at = Instant::now();
    let output = fixture
        .command()
        .env("RDBG_FAKE_STDOUT", remote_stdout)
        .env("RDBG_FAKE_STDERR", remote_stderr)
        .env("RDBG_FAKE_SLEEP", "30")
        .args([
            "--json",
            "--timeout",
            "3",
            "ps",
            "Z3",
            "--eval",
            "Start-Sleep 30",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_EXECUTION_TIMEOUT));
    assert!(
        started_at.elapsed() < Duration::from_secs(10),
        "timeout command took {:?}",
        started_at.elapsed()
    );
    let result = parse_json(&output);
    assert_eq!(result["ok"], false);
    assert_eq!(result["error_class"], "execution_timeout");
    assert_eq!(result["rdbg_exit_code"], rdbg::EXIT_EXECUTION_TIMEOUT);
    assert_eq!(result["timeout_seconds"], 3);
    assert!(result["process_exit_code"].is_null());
    assert_eq!(result["stdout"], remote_stdout);
    assert_eq!(result["stderr"], remote_stderr);
    assert_eq!(result["stdout_utf8_lossy"], false);
    assert_eq!(result["stderr_utf8_lossy"], false);
}

#[test]
fn timeout_kills_descendant_that_keeps_output_pipes_open() {
    let fixture = Fixture::new();
    let remote_stdout = "leader stdout before descendant 中文\n";
    let remote_stderr = "leader stderr before descendant\n";
    let started_at = Instant::now();
    let output = fixture
        .command()
        .env("RDBG_FAKE_STDOUT", remote_stdout)
        .env("RDBG_FAKE_STDERR", remote_stderr)
        .env("RDBG_FAKE_DESCENDANT_SLEEP", "30")
        .args([
            "--json",
            "--timeout",
            "3",
            "ps",
            "Z3",
            "--eval",
            "Write-Output descendant",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_EXECUTION_TIMEOUT));
    assert!(
        started_at.elapsed() < Duration::from_secs(10),
        "process-tree timeout took {:?}",
        started_at.elapsed()
    );
    let result = parse_json(&output);
    assert_eq!(result["error_class"], "execution_timeout");
    assert_eq!(result["timeout_seconds"], 3);
    assert!(result["process_exit_code"].is_null());
    assert_eq!(result["stdout"], remote_stdout);
    assert_eq!(result["stderr"], remote_stderr);
}

#[test]
fn json_capture_is_bounded_per_stream_and_reports_truncation() {
    let fixture = Fixture::new();
    let requested_bytes = rdbg::MAX_CAPTURE_BYTES_PER_STREAM + 32 * 1024;
    let output = fixture
        .command()
        .env("RDBG_FAKE_STDOUT_BYTES", requested_bytes.to_string())
        .env("RDBG_FAKE_STDERR_BYTES", requested_bytes.to_string())
        .args(["--json", "ps", "Z3", "--eval", "Write-Output large"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_OK));
    let result = parse_json(&output);
    assert_eq!(
        result["stdout"].as_str().unwrap().len(),
        rdbg::MAX_CAPTURE_BYTES_PER_STREAM
    );
    assert_eq!(
        result["stderr"].as_str().unwrap().len(),
        rdbg::MAX_CAPTURE_BYTES_PER_STREAM
    );
    assert_eq!(result["stdout_truncated"], true);
    assert_eq!(result["stderr_truncated"], true);
}

#[test]
fn timeout_requires_a_strict_positive_integer_within_the_limit() {
    let invalid_values = ["", "0", "-1", "+1", "1.5", "abc", "86401"];

    for value in invalid_values {
        let output = Command::new(env!("CARGO_BIN_EXE_rdbg"))
            .args([
                "--json",
                "--timeout",
                value,
                "ps",
                "Z3",
                "--eval",
                "Write-Output ok",
            ])
            .output()
            .unwrap();

        assert_eq!(output.status.code(), Some(rdbg::EXIT_USAGE), "{value:?}");
        let result = parse_json(&output);
        assert_eq!(result["error_class"], "usage", "{value:?}");
        assert_eq!(result["rdbg_exit_code"], rdbg::EXIT_USAGE, "{value:?}");
        assert!(
            result["message"]
                .as_str()
                .unwrap()
                .contains("integer from 1 to 86400"),
            "{value:?}: {}",
            result["message"]
        );
    }

    let output = Command::new(env!("CARGO_BIN_EXE_rdbg"))
        .args(["--json", "--timeout"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(rdbg::EXIT_USAGE));
    let result = parse_json(&output);
    assert_eq!(result["error_class"], "usage");
    assert!(
        result["message"]
            .as_str()
            .unwrap()
            .contains("requires SECONDS")
    );
}

#[test]
fn ssh_start_failure_has_stable_json_error() {
    let missing = format!(
        "/definitely/missing/rdbg-ssh-{}",
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    );
    let output = Command::new(env!("CARGO_BIN_EXE_rdbg"))
        .args([
            "--json",
            "--timeout",
            "3",
            "--ssh",
            &missing,
            "ps",
            "Z3",
            "--eval",
            "Write-Output ok",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_SSH_LAUNCH));
    let result = parse_json(&output);
    assert_eq!(result["error_class"], "ssh_launch");
    assert_eq!(result["rdbg_exit_code"], rdbg::EXIT_SSH_LAUNCH);
    assert_eq!(result["timeout_seconds"], 3);
    assert!(result["process_exit_code"].is_null());
}

#[test]
fn exit_255_transport_stderr_patterns_have_specific_classes() {
    let cases = [
        ("websocket: bad handshake\n", "cloudflared_handshake"),
        ("Host key verification failed.\n", "ssh_hostkey"),
        ("Permission denied (publickey).\n", "ssh_auth"),
        (
            "ssh: connect to host Z3 port 22: Operation timed out\n",
            "ssh_timeout",
        ),
        (
            "ssh: Could not resolve hostname Z3: nodename nor servname provided\n",
            "dns_resolution",
        ),
        (
            "ssh: connect to host Z3 port 22: Connection refused\n",
            "connection_refused",
        ),
        (
            "kex_exchange_identification: Connection closed\n",
            "ssh_transport",
        ),
    ];

    for (stderr, expected_class) in cases {
        let fixture = Fixture::new();
        let output = fixture
            .command()
            .env("RDBG_FAKE_STDERR", stderr)
            .env("RDBG_FAKE_EXIT", "255")
            .args(["--json", "ps", "Z3", "--eval", "Write-Output ok"])
            .output()
            .unwrap();

        assert_eq!(output.status.code(), Some(rdbg::EXIT_SSH_TRANSPORT));
        let result = parse_json(&output);
        assert_eq!(result["error_class"], expected_class, "stderr={stderr:?}");
        assert_eq!(result["rdbg_exit_code"], rdbg::EXIT_SSH_TRANSPORT);
        assert_eq!(result["process_exit_code"], 255);
        assert_eq!(result["stderr"], stderr);
    }
}

#[test]
fn file_path_error_has_stable_json_error() {
    let missing = Path::new("/definitely/missing/rdbg script.ps1");
    let output = Command::new(env!("CARGO_BIN_EXE_rdbg"))
        .arg("--json")
        .arg("ps")
        .arg("Z3")
        .arg("--file")
        .arg(missing)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_LOCAL_INPUT));
    let result = parse_json(&output);
    assert_eq!(result["error_class"], "local_input");
    assert_eq!(result["rdbg_exit_code"], rdbg::EXIT_LOCAL_INPUT);
}

#[test]
fn no_ssh_opts_skips_hardening_options() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args(["--no-ssh-opts", "ps", "Z3", "--eval", "Write-Output ok"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_OK));
    let argv = fixture.argv();
    assert_bare_ssh_argv(&argv, "Z3");
    assert!(!argv.iter().any(|a| a == "-o"), "no -o flags: {argv:#?}");
}

#[test]
fn invalid_script_encoding_is_a_local_input_error() {
    let fixture = Fixture::new();
    let script_path = fixture.root.join("invalid utf8.ps1");
    fs::write(&script_path, [0x80, 0x81, 0x82]).unwrap();
    let output = fixture
        .command()
        .arg("--json")
        .arg("ps")
        .arg("Z3")
        .arg("--file")
        .arg(&script_path)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(rdbg::EXIT_LOCAL_INPUT));
    let result = parse_json(&output);
    assert_eq!(result["error_class"], "local_input");
    assert!(
        result["message"]
            .as_str()
            .unwrap()
            .contains("not valid UTF-8")
    );
    assert!(
        !fixture.argv_log.exists(),
        "ssh should not have been launched"
    );
}
