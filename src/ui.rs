use comfy_table::{
    modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Attribute, Cell, ContentArrangement, Table,
};
use console::{measure_text_width, style};

pub const MIN_PANEL_WIDTH: usize = 48;
pub const DEFAULT_WRAP_WIDTH: usize = 104;

/// Renders a labelled, non-interactive TUI-style panel for human CLI output.
pub struct Panel {
    title: String,
    rows: Vec<String>,
}

impl Panel {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            rows: Vec::new(),
        }
    }

    pub fn push(&mut self, row: impl Into<String>) {
        self.rows.push(row.into());
    }

    pub fn kv(&mut self, label: &str, value: impl AsRef<str>, width: usize) {
        self.push(kv_row(label, value.as_ref(), width));
    }

    pub fn status(
        &mut self,
        label: &str,
        healthy: bool,
        ok_text: &str,
        err_text: &str,
        width: usize,
    ) {
        let value = if healthy { ok(ok_text) } else { err(err_text) };
        self.kv(label, value, width);
    }

    pub fn prose(&mut self, label: &str, text: &str, label_width: usize) {
        for row in wrapped_dim_rows(label, text, label_width, DEFAULT_WRAP_WIDTH) {
            self.push(row);
        }
    }

    pub fn hint(&mut self, hint: &str) {
        for row in wrapped_hint_rows(hint, DEFAULT_WRAP_WIDTH) {
            self.push(row);
        }
    }

    pub fn render(&self) {
        let title_visual = measure_text_width(&self.title);
        let max_row_visual = self
            .rows
            .iter()
            .map(|row| measure_text_width(row))
            .max()
            .unwrap_or(0);
        let box_width = (max_row_visual + 4)
            .max(title_visual + 6)
            .max(MIN_PANEL_WIDTH);
        let inner_width = box_width - 4;
        let top_fill = box_width.saturating_sub(title_visual + 5);

        println!(
            "╭─ {} {}╮",
            style(&self.title).bold().cyan(),
            "─".repeat(top_fill)
        );
        for row in &self.rows {
            let visual = measure_text_width(row);
            let pad = inner_width.saturating_sub(visual);
            println!("│ {}{} │", row, " ".repeat(pad));
        }
        println!("╰{}╯", "─".repeat(box_width - 2));
    }
}

pub fn command_header(name: &str, suffix: impl AsRef<str>) {
    let suffix = suffix.as_ref();
    if suffix.is_empty() {
        println!("{}", style(name).bold().cyan());
    } else {
        println!("{}  {}", style(name).bold().cyan(), style(suffix).dim());
    }
    println!();
}

pub fn kv_row(label: &str, value: &str, width: usize) -> String {
    format!("  {}  {}", style(format!("{label:<width$}")).dim(), value)
}

pub fn ok(text: &str) -> String {
    format!("{} {}", style("✓").green().bold(), style(text).green())
}

pub fn err(text: &str) -> String {
    format!("{} {}", style("✗").red().bold(), style(text).red())
}

pub fn warn(text: &str) -> String {
    format!("{} {}", style("!").yellow().bold(), style(text).yellow())
}

pub fn wrapped_dim_rows(
    label: &str,
    text: &str,
    label_width: usize,
    value_width: usize,
) -> Vec<String> {
    wrap_words(text, value_width)
        .into_iter()
        .enumerate()
        .map(|(idx, line)| {
            let label = if idx == 0 { label } else { "" };
            format!(
                "  {}  {}",
                style(format!("{label:<label_width$}")).dim(),
                style(line).dim()
            )
        })
        .collect()
}

pub fn wrapped_hint_rows(hint: &str, width: usize) -> Vec<String> {
    wrap_words(hint, width)
        .into_iter()
        .enumerate()
        .map(|(idx, line)| {
            if idx == 0 {
                format!("  {} {}", style("→").yellow().bold(), style(line).yellow())
            } else {
                format!("    {}", style(line).yellow())
            }
        })
        .collect()
}

pub fn wrap_words(text: &str, max_width: usize) -> Vec<String> {
    let options = textwrap::Options::new(max_width)
        .break_words(false)
        .word_separator(textwrap::WordSeparator::AsciiSpace)
        .word_splitter(textwrap::WordSplitter::NoHyphenation);
    let wrapped = textwrap::wrap(text, options);
    if wrapped.is_empty() {
        vec![String::new()]
    } else {
        wrapped.into_iter().map(|line| line.into_owned()).collect()
    }
}

pub fn render_table(title: &str, headers: &[&str], rows: Vec<Vec<String>>) {
    println!("{}", style(title).bold().cyan());
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(
            headers
                .iter()
                .map(|header| Cell::new(*header).add_attribute(Attribute::Bold))
                .collect::<Vec<_>>(),
        );

    for row in rows {
        table.add_row(row);
    }

    println!("{table}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_words_wraps_long_text() {
        let rows = wrap_words("alpha beta gamma delta", 12);
        assert_eq!(rows, vec!["alpha beta", "gamma delta"]);
    }

    #[test]
    fn test_kv_row_pads_before_styling() {
        let row = kv_row("key", "value", 6);
        assert!(row.contains("key"));
        assert!(row.contains("value"));
    }
}
