use crate::types::SourceKind;
use std::io::IsTerminal;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

const SOURCE_COUNT: usize = 3;
const BAR_WIDTH: usize = 28;
const BOX_WIDTH: usize = 66;
const LINE_COUNT: usize = 10;

pub struct Progress {
    totals_bytes: [u64; SOURCE_COUNT],
    parsed_bytes: [AtomicU64; SOURCE_COUNT],
    files_total: [u64; SOURCE_COUNT],
    files_done: [AtomicU64; SOURCE_COUNT],
    produced: [AtomicU64; SOURCE_COUNT],
    indexed: [AtomicU64; SOURCE_COUNT],
    embedded: [AtomicU64; SOURCE_COUNT],
    embed_pending: [AtomicU64; SOURCE_COUNT],
    embed_total: [AtomicU64; SOURCE_COUNT],
    embed_ready: AtomicBool,
    done: AtomicBool,
    embeddings: bool,
}

impl Progress {
    pub fn new(totals_bytes: [u64; SOURCE_COUNT], files_total: [u64; SOURCE_COUNT], embeddings: bool) -> Self {
        Self {
            totals_bytes,
            parsed_bytes: std::array::from_fn(|_| AtomicU64::new(0)),
            files_total,
            files_done: std::array::from_fn(|_| AtomicU64::new(0)),
            produced: std::array::from_fn(|_| AtomicU64::new(0)),
            indexed: std::array::from_fn(|_| AtomicU64::new(0)),
            embedded: std::array::from_fn(|_| AtomicU64::new(0)),
            embed_pending: std::array::from_fn(|_| AtomicU64::new(0)),
            embed_total: std::array::from_fn(|_| AtomicU64::new(0)),
            embed_ready: AtomicBool::new(false),
            done: AtomicBool::new(false),
            embeddings,
        }
    }

    pub fn add_parsed_bytes(&self, source: SourceKind, bytes: u64) {
        self.parsed_bytes[source.idx()].fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn add_files_done(&self, source: SourceKind, count: u64) {
        self.files_done[source.idx()].fetch_add(count, Ordering::Relaxed);
    }

    pub fn add_produced(&self, source: SourceKind, count: u64) {
        self.produced[source.idx()].fetch_add(count, Ordering::Relaxed);
    }

    pub fn add_indexed(&self, source: SourceKind, count: u64) {
        self.indexed[source.idx()].fetch_add(count, Ordering::Relaxed);
    }

    pub fn add_embed_total(&self, source: SourceKind, count: u64) {
        self.embed_total[source.idx()].fetch_add(count, Ordering::Relaxed);
    }

    pub fn add_embed_pending(&self, source: SourceKind, count: u64) {
        self.embed_pending[source.idx()].fetch_add(count, Ordering::Relaxed);
    }

    pub fn sub_embed_pending(&self, source: SourceKind, count: u64) {
        self.embed_pending[source.idx()].fetch_sub(count, Ordering::Relaxed);
    }

    pub fn add_embedded(&self, source: SourceKind, count: u64) {
        self.embedded[source.idx()].fetch_add(count, Ordering::Relaxed);
    }

    pub fn set_embed_ready(&self) {
        self.embed_ready.store(true, Ordering::Relaxed);
    }

    pub fn finish(&self) {
        self.done.store(true, Ordering::SeqCst);
    }
}

pub fn spawn_reporter(progress: Arc<Progress>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut stderr = std::io::stderr();
        if !stderr.is_terminal() {
            while !progress.done.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(200));
            }
            return;
        }

        let _ = write!(stderr, "\x1b[?25l");
        for _ in 0..LINE_COUNT {
            let _ = writeln!(stderr);
        }
        let _ = stderr.flush();

        let mut last_lines = vec![String::new(); LINE_COUNT];
        let mut tick: u64 = 0;
        loop {
            let done = progress.done.load(Ordering::Relaxed);
            let lines = format_lines(&progress, tick);
            if lines != last_lines {
                let _ = write!(stderr, "\x1b[{}A", LINE_COUNT);
                for line in lines.iter() {
                    let _ = write!(stderr, "\x1b[2K{line}\n");
                }
                let _ = stderr.flush();
                last_lines = lines;
            }
            if done {
                let _ = write!(stderr, "\x1b[?25h");
                let _ = stderr.flush();
                break;
            }
            tick = tick.wrapping_add(1);
            thread::sleep(Duration::from_millis(200));
        }
    })
}

fn format_lines(progress: &Progress, tick: u64) -> Vec<String> {
    let stats = snapshot(progress);
    let claude = stats.claude;
    let codex = stats.codex;

    let mut lines = Vec::with_capacity(LINE_COUNT);
    lines.extend(render_box("claude", &claude, tick));
    lines.extend(render_box("codex", &codex, tick));
    lines
}

struct SourceStats {
    parsed: u64,
    total: u64,
    files_done: u64,
    files_total: u64,
    produced: u64,
    indexed: u64,
    embedded: u64,
    embed_total: u64,
    pending: u64,
    embeddings_enabled: bool,
    embed_ready: bool,
}

struct Snapshot {
    claude: SourceStats,
    codex: SourceStats,
}

fn snapshot(progress: &Progress) -> Snapshot {
    let parsed = load_arr(&progress.parsed_bytes);
    let produced = load_arr(&progress.produced);
    let indexed = load_arr(&progress.indexed);
    let embedded = load_arr(&progress.embedded);
    let pending = load_arr(&progress.embed_pending);
    let embed_total = load_arr(&progress.embed_total);
    let files_done = load_arr(&progress.files_done);

    let claude = SourceStats {
        parsed: parsed[SourceKind::Claude.idx()],
        total: progress.totals_bytes[SourceKind::Claude.idx()],
        files_done: files_done[SourceKind::Claude.idx()],
        files_total: progress.files_total[SourceKind::Claude.idx()],
        produced: produced[SourceKind::Claude.idx()],
        indexed: indexed[SourceKind::Claude.idx()],
        embedded: embedded[SourceKind::Claude.idx()],
        embed_total: embed_total[SourceKind::Claude.idx()],
        pending: pending[SourceKind::Claude.idx()],
        embeddings_enabled: progress.embeddings,
        embed_ready: progress.embed_ready.load(Ordering::Relaxed),
    };

    let codex = SourceStats {
        parsed: parsed[SourceKind::CodexSession.idx()] + parsed[SourceKind::CodexHistory.idx()],
        total: progress.totals_bytes[SourceKind::CodexSession.idx()]
            + progress.totals_bytes[SourceKind::CodexHistory.idx()],
        files_done: files_done[SourceKind::CodexSession.idx()]
            + files_done[SourceKind::CodexHistory.idx()],
        files_total: progress.files_total[SourceKind::CodexSession.idx()]
            + progress.files_total[SourceKind::CodexHistory.idx()],
        produced: produced[SourceKind::CodexSession.idx()] + produced[SourceKind::CodexHistory.idx()],
        indexed: indexed[SourceKind::CodexSession.idx()] + indexed[SourceKind::CodexHistory.idx()],
        embedded: embedded[SourceKind::CodexSession.idx()] + embedded[SourceKind::CodexHistory.idx()],
        embed_total: embed_total[SourceKind::CodexSession.idx()]
            + embed_total[SourceKind::CodexHistory.idx()],
        pending: pending[SourceKind::CodexSession.idx()] + pending[SourceKind::CodexHistory.idx()],
        embeddings_enabled: progress.embeddings,
        embed_ready: progress.embed_ready.load(Ordering::Relaxed),
    };

    Snapshot { claude, codex }
}

fn render_box(title: &str, stats: &SourceStats, tick: u64) -> Vec<String> {
    let inner = BOX_WIDTH - 2;
    let mut lines = Vec::with_capacity(5);

    let top_fill = inner.saturating_sub(title.len() + 2);
    lines.push(format!("┌ {title} {}┐", "─".repeat(top_fill)));

    let parse_pct = percent(stats.parsed, stats.total);
    let parse_bar = bar(parse_pct);
    let parse_bytes = format_bytes_progress(stats.parsed, stats.total);
    let parse_text = if stats.files_total > 0 {
        format!(
            "parse  {}  {:>3}% {} f{}/{}",
            parse_bar, parse_pct, parse_bytes, stats.files_done, stats.files_total
        )
    } else {
        format!(
            "parse  {}  {:>3}% {}",
            parse_bar, parse_pct, parse_bytes
        )
    };
    lines.push(format!("│ {} │", pad(parse_text, inner - 2)));

    let index_known = parse_pct == 100 && stats.produced > 0;
    let index_bar = if index_known {
        bar(percent(stats.indexed, stats.produced))
    } else {
        indeterminate_bar(tick)
    };
    let index_text = if index_known {
        format!(
            "index  {}  {} rec",
            index_bar,
            format_count_commas(stats.indexed)
        )
    } else {
        format!(
            "index  {}  {} rec",
            index_bar,
            format_count_commas(stats.indexed)
        )
    };
    lines.push(format!("│ {} │", pad(index_text, inner - 2)));

    let embed_known = parse_pct == 100 && stats.embed_total > 0;
    let embed_bar = if embed_known {
        bar(percent(stats.embedded, stats.embed_total))
    } else {
        indeterminate_bar(tick.wrapping_add(7))
    };
    let mut embed_text = format!(
        "embed  {}  {} emb",
        embed_bar,
        format_count_commas(stats.embedded)
    );
    if stats.pending > 0 {
        embed_text.push_str(&format!(" processing {}", format_count_commas(stats.pending)));
    }
    if stats.embeddings_enabled && !stats.embed_ready {
        embed_text.push_str(" init");
    }
    if !stats.embeddings_enabled {
        embed_text.push_str(" (off)");
    }
    lines.push(format!("│ {} │", pad(embed_text, inner - 2)));

    lines.push(format!("└{}┘", "─".repeat(inner)));
    lines
}

fn bar(percent: u64) -> String {
    let filled = ((percent as usize) * BAR_WIDTH / 100).min(BAR_WIDTH);
    let empty = BAR_WIDTH.saturating_sub(filled);
    format!("{}{}", "━".repeat(filled), "░".repeat(empty))
}

fn indeterminate_bar(tick: u64) -> String {
    let block = 6usize.min(BAR_WIDTH);
    let span = BAR_WIDTH.saturating_sub(block);
    let offset = if span == 0 { 0 } else { (tick as usize) % (span + 1) };
    let mut out = String::with_capacity(BAR_WIDTH);
    for i in 0..BAR_WIDTH {
        if i >= offset && i < offset + block {
            out.push('━');
        } else {
            out.push('░');
        }
    }
    out
}

fn percent(done: u64, total: u64) -> u64 {
    if total == 0 {
        return 0;
    }
    (done.saturating_mul(100) / total).min(100)
}

fn pad(input: String, width: usize) -> String {
    let mut out = String::new();
    let mut count = 0usize;
    for ch in input.chars() {
        if count >= width {
            break;
        }
        out.push(ch);
        count += 1;
    }
    if count < width {
        out.extend(std::iter::repeat(' ').take(width - count));
    }
    out
}

fn format_bytes_progress(done: u64, total: u64) -> String {
    if total == 0 {
        return "0 B".to_string();
    }
    let (done_val, done_unit) = format_bytes_parts(done);
    let (total_val, total_unit) = format_bytes_parts(total);
    if done >= total {
        return format!("{total_val} {total_unit}");
    }
    if done_unit == total_unit {
        format!("{done_val}/{total_val} {total_unit}")
    } else {
        format!("{done_val} {done_unit}/{total_val} {total_unit}")
    }
}

fn format_bytes_parts(value: u64) -> (String, &'static str) {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let v = value as f64;
    if v >= GB {
        (format!("{:.1}", v / GB), "GB")
    } else if v >= MB {
        (format!("{:.1}", v / MB), "MB")
    } else if v >= KB {
        (format!("{:.1}", v / KB), "KB")
    } else {
        (value.to_string(), "B")
    }
}

fn format_count_commas(value: u64) -> String {
    let s = value.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn load_arr(arr: &[AtomicU64; SOURCE_COUNT]) -> [u64; SOURCE_COUNT] {
    [
        arr[0].load(Ordering::Relaxed),
        arr[1].load(Ordering::Relaxed),
        arr[2].load(Ordering::Relaxed),
    ]
}
