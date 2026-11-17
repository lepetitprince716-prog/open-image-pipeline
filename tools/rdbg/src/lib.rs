use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt as _;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::json;

pub const EXIT_OK: i32 = 0;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_LOCAL_INPUT: i32 = 10;
pub const EXIT_SSH_LAUNCH: i32 = 20;
pub const EXIT_SSH_IO: i32 = 21;
pub const EXIT_SSH_TRANSPORT: i32 = 22;
pub const EXIT_REMOTE_COMMAND: i32 = 23;
pub const EXIT_SSH_TERMINATED: i32 = 24;
pub const EXIT_EXECUTION_TIMEOUT: i32 = 25;
pub const MAX_TIMEOUT_SECONDS: u64 = 86_400;
pub const MAX_CAPTURE_BYTES_PER_STREAM: usize = 1024 * 1024;

const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const TERMINATION_GRACE: Duration = Duration::from_millis(250);
const PIPE_READ_CHUNK_BYTES: usize = 16 * 1024;

/// This command is intentionally static and small. Dynamic script text is sent as an
/// ASCII Base64 payload on stdin, so neither local shell quoting nor the Windows command
/// line length limit can alter it.
pub const POWERSHELL_BOOTSTRAP: &str = r#"$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
try {
    $payload=[Console]::In.ReadToEnd()
    $bytes=[Convert]::FromBase64String($payload)
    $text=[Text.Encoding]::Unicode.GetString($bytes)
    & ([ScriptBlock]::Create($text))
} catch {
    [Console]::Error.WriteLine('PowerShell error: ' + $_.Exception.Message)
    if ($_.InvocationInfo.PositionMessage) {
        [Console]::Error.WriteLine($_.InvocationInfo.PositionMessage)
    }
    exit 1
}"#;

pub const HELP: &str = r#"rdbg - reliable remote PowerShell over system ssh

Usage:
  rdbg [--json] [--ssh SSH] [--no-ssh-opts] [--timeout SECONDS] ps TARGET --eval SCRIPT
  rdbg [--json] [--ssh SSH] [--no-ssh-opts] [--timeout SECONDS] ps TARGET --file PATH
  rdbg [--json] [--ssh SSH] [--no-ssh-opts] [--timeout SECONDS] ps TARGET --file -

Options:
  --json             Emit one JSON result object instead of human-oriented output.
  --ssh SSH          ssh executable name or path (default: ssh).
  --no-ssh-opts      Skip injecting -o SSH options (BatchMode, StrictHostKeyChecking,
                     ConnectTimeout, ServerAlive*). Use when --ssh points to a managed
                     wrapper that already handles these.
  --timeout SECONDS  Total ssh execution timeout (1-86400); default: no timeout.
  -h, --help         Show this help.

TARGET may be a host, user@host, or an alias from ssh_config.
"#;

#[derive(Debug)]
struct Config {
    json: bool,
    ssh: OsString,
    no_ssh_opts: bool,
    timeout_seconds: Option<u64>,
    target: String,
    source: ScriptSource,
}

#[derive(Debug)]
enum ScriptSource {
    Eval(String),
    File(PathBuf),
    Stdin,
}

impl ScriptSource {
    fn mode(&self) -> &'static str {
        match self {
            Self::Eval(_) => "eval",
            Self::File(_) | Self::Stdin => "file",
        }
    }
}

#[derive(Debug)]
enum ParseResult {
    Help,
    Run(Config),
}

#[derive(Debug)]
struct Failure {
    class: &'static str,
    rdbg_exit_code: i32,
    message: String,
    target: Option<String>,
    mode: Option<&'static str>,
    timeout_seconds: Option<u64>,
}

#[derive(Debug)]
struct ProcessResult {
    target: String,
    mode: &'static str,
    timeout_seconds: Option<u64>,
    timed_out: bool,
    process_exit_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_truncated: bool,
    stderr_truncated: bool,
    process_error: Option<String>,
    io_errors: Vec<String>,
}

#[derive(Debug)]
struct WaitOutcome {
    status: Option<ExitStatus>,
    timed_out: bool,
    error: Option<String>,
}

#[derive(Debug, Default)]
struct BoundedCapture {
    bytes: Vec<u8>,
    truncated: bool,
}

#[derive(Debug)]
enum WorkerEvent {
    Stdin(Result<(), String>),
    Stdout(Result<(), String>),
    Stderr(Result<(), String>),
}

#[derive(Debug, Default)]
struct WorkerState {
    stdin_done: bool,
    stdout_done: bool,
    stderr_done: bool,
    io_errors: Vec<String>,
}

impl WorkerState {
    fn all_done(&self) -> bool {
        self.stdin_done && self.stdout_done && self.stderr_done
    }

    fn record(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::Stdin(result) => {
                self.stdin_done = true;
                if let Err(error) = result {
                    self.io_errors
                        .push(format!("failed to send script payload to ssh: {error}"));
                }
            }
            WorkerEvent::Stdout(result) => {
                self.stdout_done = true;
                if let Err(error) = result {
                    self.io_errors
                        .push(format!("failed to read ssh stdout: {error}"));
                }
            }
            WorkerEvent::Stderr(result) => {
                self.stderr_done = true;
                if let Err(error) = result {
                    self.io_errors
                        .push(format!("failed to read ssh stderr: {error}"));
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ProcessTree {
    #[cfg(unix)]
    process_group_id: Option<libc::pid_t>,
}

impl Failure {
    fn simple(class: &'static str, rdbg_exit_code: i32, message: impl Into<String>) -> Self {
        Self {
            class,
            rdbg_exit_code,
            message: message.into(),
            target: None,
            mode: None,
            timeout_seconds: None,
        }
    }
}

pub fn main_entry() -> i32 {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let wants_json = args.iter().any(|arg| arg == "--json");

    let config = match parse_args(&args) {
        Ok(ParseResult::Help) => {
            let _ = io::stdout().write_all(HELP.as_bytes());
            return EXIT_OK;
        }
        Ok(ParseResult::Run(config)) => config,
        Err(failure) => {
            render_failure(&failure, wants_json);
            return failure.rdbg_exit_code;
        }
    };

    let script = match load_script(&config.source) {
        Ok(script) => script,
        Err(message) => {
            let failure = Failure {
                class: "local_input",
                rdbg_exit_code: EXIT_LOCAL_INPUT,
                message,
                target: Some(config.target.clone()),
                mode: Some(config.source.mode()),
                timeout_seconds: config.timeout_seconds,
            };
            render_failure(&failure, config.json);
            return failure.rdbg_exit_code;
        }
    };

    match execute(&config, &script) {
        Ok(result) => render_process_result(&result, config.json),
        Err(failure) => {
            render_failure(&failure, config.json);
            failure.rdbg_exit_code
        }
    }
}

fn parse_args(args: &[OsString]) -> Result<ParseResult, Failure> {
    let mut json = false;
    let mut ssh = OsString::from("ssh");
    let mut no_ssh_opts = false;
    let mut timeout_seconds = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].to_str() {
            Some("--json") => {
                json = true;
                index += 1;
            }
            Some("--no-ssh-opts") => {
                no_ssh_opts = true;
                index += 1;
            }
            Some("--ssh") => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage_failure("--ssh requires an executable name or path"))?;
                if value.is_empty() {
                    return Err(usage_failure("--ssh value must not be empty"));
                }
                ssh = value.clone();
                index += 2;
            }
            Some("--timeout") => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage_failure("--timeout requires SECONDS"))?;
                timeout_seconds = Some(parse_timeout_seconds(value)?);
                index += 2;
            }
            Some("-h" | "--help") => return Ok(ParseResult::Help),
            Some("ps") => break,
            Some(value) if value.starts_with('-') => {
                return Err(usage_failure(format!("unknown option: {value}")));
            }
            Some(value) => {
                return Err(usage_failure(format!(
                    "expected subcommand 'ps', found: {value}"
                )));
            }
            None => return Err(usage_failure("arguments must be valid Unicode")),
        }
    }

    if index >= args.len() {
        return Err(usage_failure("missing subcommand 'ps'"));
    }
    index += 1;

    let target = unicode_arg(args.get(index), "missing TARGET after 'ps'")?;
    if target.is_empty() {
        return Err(usage_failure("TARGET must not be empty"));
    }
    if target.starts_with('-') {
        return Err(usage_failure("TARGET must not start with '-'"));
    }
    index += 1;

    let mode = unicode_arg(
        args.get(index),
        "missing --eval SCRIPT or --file PATH after TARGET",
    )?;
    index += 1;

    let value = args
        .get(index)
        .ok_or_else(|| usage_failure(format!("{mode} requires a value")))?;
    index += 1;

    if index != args.len() {
        return Err(usage_failure("unexpected extra arguments"));
    }

    let source = match mode.as_str() {
        "--eval" => ScriptSource::Eval(
            value
                .to_str()
                .ok_or_else(|| usage_failure("SCRIPT must be valid Unicode"))?
                .to_owned(),
        ),
        "--file" if value == "-" => ScriptSource::Stdin,
        "--file" => ScriptSource::File(PathBuf::from(value)),
        _ => {
            return Err(usage_failure(format!(
                "expected --eval or --file, found: {mode}"
            )));
        }
    };

    Ok(ParseResult::Run(Config {
        json,
        ssh,
        no_ssh_opts,
        timeout_seconds,
        target,
        source,
    }))
}

fn parse_timeout_seconds(value: &OsString) -> Result<u64, Failure> {
    let value = value
        .to_str()
        .ok_or_else(|| usage_failure("--timeout SECONDS must be valid Unicode"))?;
    let invalid = || {
        usage_failure(format!(
            "--timeout SECONDS must be an integer from 1 to {MAX_TIMEOUT_SECONDS}"
        ))
    };

    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }

    let seconds = value.parse::<u64>().map_err(|_| invalid())?;
    if !(1..=MAX_TIMEOUT_SECONDS).contains(&seconds) {
        return Err(invalid());
    }

    Ok(seconds)
}

fn unicode_arg(arg: Option<&OsString>, missing: &str) -> Result<String, Failure> {
    let arg = arg.ok_or_else(|| usage_failure(missing))?;
    arg.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| usage_failure("arguments must be valid Unicode"))
}

fn usage_failure(message: impl Into<String>) -> Failure {
    Failure::simple("usage", EXIT_USAGE, message)
}

fn load_script(source: &ScriptSource) -> Result<String, String> {
    match source {
        ScriptSource::Eval(script) => Ok(script.clone()),
        ScriptSource::File(path) => {
            let bytes = fs::read(path)
                .map_err(|error| format!("failed to read '{}': {error}", path.display()))?;
            decode_script(bytes, &format!("'{}'", path.display()))
        }
        ScriptSource::Stdin => {
            let mut bytes = Vec::new();
            io::stdin()
                .read_to_end(&mut bytes)
                .map_err(|error| format!("failed to read script from stdin: {error}"))?;
            decode_script(bytes, "stdin")
        }
    }
}

fn decode_script(bytes: Vec<u8>, source: &str) -> Result<String, String> {
    if let Some(bytes) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        return String::from_utf8(bytes.to_vec())
            .map_err(|error| format!("script from {source} is not valid UTF-8: {error}"));
    }

    if let Some(bytes) = bytes.strip_prefix(&[0xff, 0xfe]) {
        return decode_utf16(bytes, true, source);
    }

    if let Some(bytes) = bytes.strip_prefix(&[0xfe, 0xff]) {
        return decode_utf16(bytes, false, source);
    }

    String::from_utf8(bytes)
        .map_err(|error| format!("script from {source} is not valid UTF-8: {error}"))
}

fn decode_utf16(bytes: &[u8], little_endian: bool, source: &str) -> Result<String, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "UTF-16 script from {source} has an odd byte length"
        ));
    }

    let units = bytes.chunks_exact(2).map(|pair| {
        if little_endian {
            u16::from_le_bytes([pair[0], pair[1]])
        } else {
            u16::from_be_bytes([pair[0], pair[1]])
        }
    });
    std::char::decode_utf16(units)
        .collect::<Result<String, _>>()
        .map_err(|error| format!("script from {source} contains invalid UTF-16: {error}"))
}

fn utf16le_base64(text: &str) -> String {
    let mut bytes = Vec::with_capacity(text.len() * 2);
    for unit in text.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    BASE64.encode(bytes)
}

fn execute(config: &Config, script: &str) -> Result<ProcessResult, Failure> {
    let bootstrap = utf16le_base64(POWERSHELL_BOOTSTRAP);
    let payload = utf16le_base64(script).into_bytes();

    let mut command = Command::new(&config.ssh);
    command.arg("-T");
    if !config.no_ssh_opts {
        command
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg("StrictHostKeyChecking=yes")
            .arg("-o")
            .arg("ConnectTimeout=10")
            .arg("-o")
            .arg("ServerAliveInterval=15")
            .arg("-o")
            .arg("ServerAliveCountMax=3");
    }
    command
        .arg(&config.target)
        .arg("powershell.exe")
        .arg("-NoLogo")
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-EncodedCommand")
        .arg(bootstrap)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // ssh and every ordinary ProxyCommand/descendant start in a dedicated
    // process group. A total deadline can therefore terminate the whole local
    // transport tree instead of only its leader.
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command.spawn().map_err(|error| Failure {
        class: "ssh_launch",
        rdbg_exit_code: EXIT_SSH_LAUNCH,
        message: format!("failed to launch {:?}: {error}", config.ssh),
        target: Some(config.target.clone()),
        mode: Some(config.source.mode()),
        timeout_seconds: config.timeout_seconds,
    })?;

    let started_at = Instant::now();
    let process_tree = ProcessTree::from_child(&child);

    let child_stdin = match child.stdin.take() {
        Some(pipe) => pipe,
        None => {
            return Err(missing_pipe_failure(
                &mut child,
                process_tree,
                config,
                "ssh child stdin was not available",
            ));
        }
    };

    let child_stdout = match child.stdout.take() {
        Some(pipe) => pipe,
        None => {
            return Err(missing_pipe_failure(
                &mut child,
                process_tree,
                config,
                "ssh child stdout was not available",
            ));
        }
    };

    let child_stderr = match child.stderr.take() {
        Some(pipe) => pipe,
        None => {
            return Err(missing_pipe_failure(
                &mut child,
                process_tree,
                config,
                "ssh child stderr was not available",
            ));
        }
    };

    let stdout_capture = Arc::new(Mutex::new(BoundedCapture::default()));
    let stderr_capture = Arc::new(Mutex::new(BoundedCapture::default()));
    let (worker_sender, worker_receiver) = mpsc::channel();
    let workers = spawn_io_workers(
        child_stdin,
        payload,
        child_stdout,
        Arc::clone(&stdout_capture),
        child_stderr,
        Arc::clone(&stderr_capture),
        worker_sender,
    );

    let mut worker_state = WorkerState::default();
    let wait = wait_for_execution(
        &mut child,
        process_tree,
        config.timeout_seconds.map(Duration::from_secs),
        started_at,
        &worker_receiver,
        &mut worker_state,
    );

    // Completed threads have sent their final event. Threads still blocked in
    // an inherited pipe after timeout are deliberately detached rather than
    // turning the configured deadline into an unbounded join.
    drop(workers);

    let (stdout, stdout_truncated) = snapshot_capture(&stdout_capture);
    let (stderr, stderr_truncated) = snapshot_capture(&stderr_capture);

    Ok(ProcessResult {
        target: config.target.clone(),
        mode: config.source.mode(),
        timeout_seconds: config.timeout_seconds,
        timed_out: wait.timed_out,
        process_exit_code: wait.status.and_then(|status| status.code()),
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
        process_error: wait.error,
        io_errors: worker_state.io_errors,
    })
}

fn write_payload(mut stdin: impl Write, payload: &[u8]) -> io::Result<()> {
    stdin.write_all(payload)?;
    stdin.flush()
}

fn missing_pipe_failure(
    child: &mut Child,
    process_tree: ProcessTree,
    config: &Config,
    message: &str,
) -> Failure {
    let (_, cleanup_error) = terminate_and_reap(
        child,
        process_tree,
        None,
        Instant::now() + TERMINATION_GRACE,
    );
    let mut message = message.to_owned();
    if let Some(cleanup_error) = cleanup_error {
        message.push_str("; ");
        message.push_str(&cleanup_error);
    }
    Failure {
        class: "ssh_io",
        rdbg_exit_code: EXIT_SSH_IO,
        message,
        target: Some(config.target.clone()),
        mode: Some(config.source.mode()),
        timeout_seconds: config.timeout_seconds,
    }
}

impl BoundedCapture {
    fn append(&mut self, chunk: &[u8]) {
        let remaining = MAX_CAPTURE_BYTES_PER_STREAM.saturating_sub(self.bytes.len());
        let retained = remaining.min(chunk.len());
        self.bytes.extend_from_slice(&chunk[..retained]);
        if retained < chunk.len() {
            self.truncated = true;
        }
    }
}

fn spawn_io_workers(
    child_stdin: impl Write + Send + 'static,
    payload: Vec<u8>,
    child_stdout: impl Read + Send + 'static,
    stdout_capture: Arc<Mutex<BoundedCapture>>,
    child_stderr: impl Read + Send + 'static,
    stderr_capture: Arc<Mutex<BoundedCapture>>,
    sender: mpsc::Sender<WorkerEvent>,
) -> Vec<JoinHandle<()>> {
    let stdin_sender = sender.clone();
    let writer = thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            write_payload(child_stdin, &payload).map_err(|error| error.to_string())
        }))
        .unwrap_or_else(|_| Err("ssh stdin writer thread panicked".to_owned()));
        let _ = stdin_sender.send(WorkerEvent::Stdin(result));
    });

    let stdout_sender = sender.clone();
    let stdout_reader = thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_pipe(child_stdout, &stdout_capture).map_err(|error| error.to_string())
        }))
        .unwrap_or_else(|_| Err("ssh stdout reader thread panicked".to_owned()));
        let _ = stdout_sender.send(WorkerEvent::Stdout(result));
    });

    let stderr_reader = thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_pipe(child_stderr, &stderr_capture).map_err(|error| error.to_string())
        }))
        .unwrap_or_else(|_| Err("ssh stderr reader thread panicked".to_owned()));
        let _ = sender.send(WorkerEvent::Stderr(result));
    });

    vec![writer, stdout_reader, stderr_reader]
}

fn read_pipe(mut pipe: impl Read, capture: &Arc<Mutex<BoundedCapture>>) -> io::Result<()> {
    let mut chunk = [0_u8; PIPE_READ_CHUNK_BYTES];
    loop {
        let read = pipe.read(&mut chunk)?;
        if read == 0 {
            return Ok(());
        }
        capture
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .append(&chunk[..read]);
    }
}

fn snapshot_capture(capture: &Arc<Mutex<BoundedCapture>>) -> (Vec<u8>, bool) {
    let capture = capture
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    (capture.bytes.clone(), capture.truncated)
}

fn drain_worker_events(receiver: &mpsc::Receiver<WorkerEvent>, worker_state: &mut WorkerState) {
    while let Ok(event) = receiver.try_recv() {
        worker_state.record(event);
    }
}

fn wait_for_execution(
    child: &mut Child,
    process_tree: ProcessTree,
    timeout: Option<Duration>,
    started_at: Instant,
    worker_receiver: &mpsc::Receiver<WorkerEvent>,
    worker_state: &mut WorkerState,
) -> WaitOutcome {
    let deadline = timeout.map(|timeout| started_at + timeout);
    let mut status = None;

    loop {
        drain_worker_events(worker_receiver, worker_state);

        if status.is_none() {
            match child.try_wait() {
                Ok(Some(child_status)) => status = Some(child_status),
                Ok(None) => {}
                Err(error) => {
                    return terminate_after_wait_error(
                        child,
                        process_tree,
                        status,
                        error,
                        worker_receiver,
                        worker_state,
                    );
                }
            }
        }

        if status.is_some() && worker_state.all_done() {
            return WaitOutcome {
                status,
                timed_out: false,
                error: None,
            };
        }

        let now = Instant::now();
        if deadline.is_some_and(|deadline| now >= deadline) {
            let cleanup_deadline = Instant::now() + TERMINATION_GRACE;
            let (_reaped_status, termination_error) =
                terminate_and_reap(child, process_tree, status, cleanup_deadline);
            drain_workers_until(worker_receiver, worker_state, cleanup_deadline);

            let mut errors = Vec::new();
            if let Some(error) = termination_error {
                errors.push(error);
            }
            record_incomplete_workers(worker_state, &mut errors);

            return WaitOutcome {
                // The timed-out command has no meaningful remote exit code.
                status: None,
                timed_out: true,
                error: (!errors.is_empty()).then(|| errors.join("; ")),
            };
        }

        let sleep_for = deadline
            .map(|deadline| deadline.saturating_duration_since(now))
            .unwrap_or(WAIT_POLL_INTERVAL)
            .min(WAIT_POLL_INTERVAL);
        thread::sleep(sleep_for);
    }
}

fn terminate_after_wait_error(
    child: &mut Child,
    process_tree: ProcessTree,
    status: Option<ExitStatus>,
    wait_error: io::Error,
    worker_receiver: &mpsc::Receiver<WorkerEvent>,
    worker_state: &mut WorkerState,
) -> WaitOutcome {
    let cleanup_deadline = Instant::now() + TERMINATION_GRACE;
    let (status, cleanup_error) = terminate_and_reap(child, process_tree, status, cleanup_deadline);
    drain_workers_until(worker_receiver, worker_state, cleanup_deadline);

    let mut error = format!("failed while waiting for ssh: {wait_error}");
    if let Some(cleanup_error) = cleanup_error {
        error.push_str("; ");
        error.push_str(&cleanup_error);
    }
    let mut incomplete_errors = Vec::new();
    record_incomplete_workers(worker_state, &mut incomplete_errors);
    for incomplete_error in incomplete_errors {
        error.push_str("; ");
        error.push_str(&incomplete_error);
    }

    WaitOutcome {
        status,
        timed_out: false,
        error: Some(error),
    }
}

fn drain_workers_until(
    receiver: &mpsc::Receiver<WorkerEvent>,
    worker_state: &mut WorkerState,
    deadline: Instant,
) {
    while !worker_state.all_done() {
        drain_worker_events(receiver, worker_state);
        if worker_state.all_done() || Instant::now() >= deadline {
            break;
        }
        thread::sleep(WAIT_POLL_INTERVAL);
    }
}

fn record_incomplete_workers(worker_state: &WorkerState, errors: &mut Vec<String>) {
    if !worker_state.stdin_done {
        errors.push("ssh stdin writer did not stop before cleanup deadline".to_owned());
    }
    if !worker_state.stdout_done {
        errors.push("ssh stdout reader did not stop before cleanup deadline".to_owned());
    }
    if !worker_state.stderr_done {
        errors.push("ssh stderr reader did not stop before cleanup deadline".to_owned());
    }
}

impl ProcessTree {
    fn from_child(child: &Child) -> Self {
        Self {
            #[cfg(unix)]
            process_group_id: libc::pid_t::try_from(child.id()).ok(),
        }
    }

    fn terminate(self, child: &mut Child) -> Option<String> {
        #[cfg(unix)]
        {
            if let Some(process_group_id) = self.process_group_id {
                // Negative pid addresses the process group. The child was
                // spawned with process_group(0), so its pid is the pgid.
                let result = unsafe { libc::kill(-process_group_id, libc::SIGKILL) };
                if result == 0 {
                    return None;
                }

                let error = io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::ESRCH) {
                    return None;
                }
                return Some(format!("failed to terminate ssh process group: {error}"));
            }
        }

        match child.kill() {
            Ok(()) => None,
            Err(error) if error.kind() == io::ErrorKind::InvalidInput => None,
            Err(error) => Some(format!("failed to terminate ssh child: {error}")),
        }
    }
}

fn terminate_and_reap(
    child: &mut Child,
    process_tree: ProcessTree,
    status: Option<ExitStatus>,
    deadline: Instant,
) -> (Option<ExitStatus>, Option<String>) {
    let termination_error = process_tree.terminate(child);
    let mut status = status;
    let mut errors = Vec::new();
    if let Some(error) = termination_error {
        errors.push(error);
    }

    while status.is_none() {
        match child.try_wait() {
            Ok(Some(child_status)) => status = Some(child_status),
            Ok(None) if Instant::now() < deadline => thread::sleep(WAIT_POLL_INTERVAL),
            Ok(None) => {
                errors.push("ssh child was not reaped before cleanup deadline".to_owned());
                break;
            }
            Err(error) => {
                errors.push(format!("failed to reap ssh child: {error}"));
                break;
            }
        }
    }

    let error = (!errors.is_empty()).then(|| errors.join("; "));
    (status, error)
}

fn render_process_result(result: &ProcessResult, as_json: bool) -> i32 {
    let (class, rdbg_exit_code, message) = classify_process_result(result);
    let ok = rdbg_exit_code == EXIT_OK;

    if as_json {
        let (stdout, stdout_lossy) = utf8_output(&result.stdout);
        let (stderr, stderr_lossy) = utf8_output(&result.stderr);
        let document = json!({
            "schema": "rdbg.result.v1",
            "ok": ok,
            "error_class": class,
            "rdbg_exit_code": rdbg_exit_code,
            "process_exit_code": result.process_exit_code,
            "target": result.target,
            "mode": result.mode,
            "timeout_seconds": result.timeout_seconds,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_utf8_lossy": stdout_lossy,
            "stderr_utf8_lossy": stderr_lossy,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "message": message,
        });
        let mut stdout_handle = io::stdout().lock();
        let _ = serde_json::to_writer(&mut stdout_handle, &document);
        let _ = stdout_handle.write_all(b"\n");
    } else {
        let _ = io::stdout().write_all(&result.stdout);
        let _ = io::stderr().write_all(&result.stderr);
        if let Some(message) = message {
            write_human_diagnostic(class.unwrap_or("ssh_io"), &message, &result.stderr);
        }
        if result.stdout_truncated || result.stderr_truncated {
            let streams = match (result.stdout_truncated, result.stderr_truncated) {
                (true, true) => "stdout and stderr",
                (true, false) => "stdout",
                (false, true) => "stderr",
                (false, false) => unreachable!(),
            };
            write_human_diagnostic(
                "output_truncated",
                &format!(
                    "{streams} exceeded the {MAX_CAPTURE_BYTES_PER_STREAM}-byte per-stream capture limit"
                ),
                &result.stderr,
            );
        }
    }

    rdbg_exit_code
}

fn classify_process_result(result: &ProcessResult) -> (Option<&'static str>, i32, Option<String>) {
    if result.timed_out {
        let seconds = result
            .timeout_seconds
            .expect("a timed-out process must have a configured timeout");
        let mut message =
            format!("ssh execution exceeded timeout of {seconds} seconds and was terminated");
        if let Some(error) = &result.process_error {
            message.push_str("; ");
            message.push_str(error);
        }
        return (
            Some("execution_timeout"),
            EXIT_EXECUTION_TIMEOUT,
            Some(message),
        );
    }

    if let Some(error) = &result.process_error {
        return (
            Some("ssh_io"),
            EXIT_SSH_IO,
            Some(with_io_errors(
                format!("ssh process handling failed: {error}"),
                result,
            )),
        );
    }

    match result.process_exit_code {
        Some(0) if result.io_errors.is_empty() => (None, EXIT_OK, None),
        Some(0) => (
            Some("ssh_io"),
            EXIT_SSH_IO,
            Some(with_io_errors("ssh pipe I/O failed".to_owned(), result)),
        ),
        Some(255) => {
            let class = classify_transport(&result.stderr);
            (
                Some(class),
                EXIT_SSH_TRANSPORT,
                Some(with_io_errors(
                    format!("ssh reported {class} (exit 255)"),
                    result,
                )),
            )
        }
        Some(code) => (
            Some("remote_command"),
            EXIT_REMOTE_COMMAND,
            Some(with_io_errors(
                format!("remote PowerShell command exited with status {code}"),
                result,
            )),
        ),
        None => (
            Some("ssh_terminated"),
            EXIT_SSH_TERMINATED,
            Some(with_io_errors(
                "ssh terminated without an exit status".to_owned(),
                result,
            )),
        ),
    }
}

fn classify_transport(stderr: &[u8]) -> &'static str {
    let stderr = String::from_utf8_lossy(stderr).to_ascii_lowercase();

    if stderr.contains("websocket: bad handshake")
        || stderr.contains("websocket bad handshake")
        || (stderr.contains("cloudflared") && stderr.contains("bad handshake"))
    {
        "cloudflared_handshake"
    } else if stderr.contains("host key verification failed")
        || stderr.contains("remote host identification has changed")
        || stderr.contains("no matching host key type found")
        || (stderr.contains("offending") && stderr.contains("host key"))
    {
        "ssh_hostkey"
    } else if stderr.contains("permission denied")
        || stderr.contains("authentication failed")
        || stderr.contains("no supported authentication methods available")
        || stderr.contains("too many authentication failures")
    {
        "ssh_auth"
    } else if stderr.contains("operation timed out")
        || stderr.contains("connection timed out")
        || stderr.contains("connect timeout")
        || stderr.contains("i/o timeout")
    {
        "ssh_timeout"
    } else if stderr.contains("could not resolve hostname")
        || stderr.contains("name or service not known")
        || stderr.contains("temporary failure in name resolution")
        || stderr.contains("nodename nor servname provided")
    {
        "dns_resolution"
    } else if stderr.contains("connection refused") {
        "connection_refused"
    } else {
        "ssh_transport"
    }
}

fn with_io_errors(mut message: String, result: &ProcessResult) -> String {
    for error in &result.io_errors {
        message.push_str("; ");
        message.push_str(error);
    }
    message
}

fn render_failure(failure: &Failure, as_json: bool) {
    if as_json {
        let document = json!({
            "schema": "rdbg.result.v1",
            "ok": false,
            "error_class": failure.class,
            "rdbg_exit_code": failure.rdbg_exit_code,
            "process_exit_code": null,
            "target": failure.target,
            "mode": failure.mode,
            "timeout_seconds": failure.timeout_seconds,
            "stdout": "",
            "stderr": "",
            "stdout_utf8_lossy": false,
            "stderr_utf8_lossy": false,
            "stdout_truncated": false,
            "stderr_truncated": false,
            "message": failure.message,
        });
        let mut stdout_handle = io::stdout().lock();
        let _ = serde_json::to_writer(&mut stdout_handle, &document);
        let _ = stdout_handle.write_all(b"\n");
    } else {
        write_human_diagnostic(failure.class, &failure.message, &[]);
        if failure.class == "usage" {
            let _ = io::stderr().write_all(b"Try 'rdbg --help'.\n");
        }
    }
}

fn write_human_diagnostic(class: &str, message: &str, existing_stderr: &[u8]) {
    let mut stderr = io::stderr().lock();
    if !existing_stderr.is_empty() && !existing_stderr.ends_with(b"\n") {
        let _ = stderr.write_all(b"\n");
    }
    let _ = writeln!(stderr, "rdbg [{class}]: {message}");
}

fn utf8_output(bytes: &[u8]) -> (String, bool) {
    match String::from_utf8(bytes.to_vec()) {
        Ok(text) => (text, false),
        Err(error) => (String::from_utf8_lossy(error.as_bytes()).into_owned(), true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16le_base64_round_trip_handles_non_bmp_unicode() {
        let input = r#"$x='中文 🙂'; Write-Output `"$x`""#;
        let encoded = utf16le_base64(input);
        let bytes = BASE64.decode(encoded).unwrap();
        let units = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]));
        let decoded = String::from_utf16(&units.collect::<Vec<_>>()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn utf16_bom_input_is_supported() {
        let input = "Write-Output '中文'";
        let mut bytes = vec![0xff, 0xfe];
        for unit in input.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        assert_eq!(decode_script(bytes, "test").unwrap(), input);
    }

    #[test]
    fn bootstrap_has_fixed_error_and_encoding_contract() {
        assert!(POWERSHELL_BOOTSTRAP.contains("$ErrorActionPreference='Stop'"));
        assert!(POWERSHELL_BOOTSTRAP.contains("UTF8Encoding]::new($false)"));
        assert!(POWERSHELL_BOOTSTRAP.contains("[Console]::In.ReadToEnd()"));
        assert!(POWERSHELL_BOOTSTRAP.contains("[Console]::Error.WriteLine"));
        assert!(POWERSHELL_BOOTSTRAP.contains("exit 1"));
    }

    #[test]
    fn transport_error_patterns_have_stable_classes() {
        let cases: &[(&[u8], &str)] = &[
            (b"websocket: bad handshake", "cloudflared_handshake"),
            (b"Host key verification failed.", "ssh_hostkey"),
            (b"Permission denied (publickey).", "ssh_auth"),
            (
                b"ssh: connect to host TARGET port 22: Operation timed out",
                "ssh_timeout",
            ),
            (
                b"ssh: Could not resolve hostname TARGET: nodename nor servname provided",
                "dns_resolution",
            ),
            (
                b"connect to host TARGET port 22: Connection refused",
                "connection_refused",
            ),
            (
                b"kex_exchange_identification: Connection closed",
                "ssh_transport",
            ),
        ];

        for (stderr, expected) in cases {
            assert_eq!(classify_transport(stderr), *expected, "{stderr:?}");
        }
    }
}
