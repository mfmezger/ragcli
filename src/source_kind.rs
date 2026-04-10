//! Shared source-kind classification for ingestion and store statistics.

use std::path::Path;

/// File kind recognized by `ragcli`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Text,
    Markdown,
    Html,
    Csv { delimiter: u8 },
    Code { language: &'static str },
    Pdf,
    Image,
    Unsupported,
}

/// Aggregated content category used for store statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentCategory {
    Text,
    Pdf,
    Image,
    Other,
}

impl SourceKind {
    /// Classifies a source path by file extension.
    pub fn from_path(path: &Path) -> Self {
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "md" | "markdown" => Self::Markdown,
            "txt" | "rst" => Self::Text,
            "html" | "htm" => Self::Html,
            "csv" => Self::Csv { delimiter: b',' },
            "tsv" => Self::Csv { delimiter: b'\t' },
            "rs" => Self::Code { language: "rust" },
            "py" => Self::Code { language: "python" },
            "js" => Self::Code {
                language: "javascript",
            },
            "ts" => Self::Code {
                language: "typescript",
            },
            "tsx" => Self::Code { language: "tsx" },
            "jsx" => Self::Code { language: "jsx" },
            "go" => Self::Code { language: "go" },
            "java" => Self::Code { language: "java" },
            "c" => Self::Code { language: "c" },
            "cc" | "cpp" | "cxx" => Self::Code { language: "cpp" },
            "h" | "hpp" => Self::Code {
                language: "c-header",
            },
            "sh" | "bash" => Self::Code { language: "shell" },
            "toml" | "yaml" | "yml" | "json" => Self::Code { language: "config" },
            "pdf" => Self::Pdf,
            "png" | "jpg" | "jpeg" | "webp" => Self::Image,
            _ => Self::Unsupported,
        }
    }

    /// Returns the metadata format label used for chunk metadata.
    pub fn format_label(self) -> Option<&'static str> {
        match self {
            Self::Text => Some("text"),
            Self::Markdown => Some("markdown"),
            Self::Html => Some("html"),
            Self::Csv { delimiter: b'\t' } => Some("tsv"),
            Self::Csv { .. } => Some("csv"),
            Self::Code { .. } => Some("code"),
            Self::Pdf => Some("pdf"),
            Self::Image => Some("image"),
            Self::Unsupported => None,
        }
    }

    /// Returns the aggregate content category used in store statistics.
    pub fn content_category(self) -> ContentCategory {
        match self {
            Self::Text | Self::Markdown | Self::Html | Self::Csv { .. } | Self::Code { .. } => {
                ContentCategory::Text
            }
            Self::Pdf => ContentCategory::Pdf,
            Self::Image => ContentCategory::Image,
            Self::Unsupported => ContentCategory::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_kind_from_path_and_content_category() {
        assert_eq!(
            SourceKind::from_path(Path::new("notes.md")),
            SourceKind::Markdown
        );
        assert_eq!(
            SourceKind::from_path(Path::new("notes.txt")),
            SourceKind::Text
        );
        assert_eq!(
            SourceKind::from_path(Path::new("page.html")),
            SourceKind::Html
        );
        assert_eq!(
            SourceKind::from_path(Path::new("table.tsv")),
            SourceKind::Csv { delimiter: b'\t' }
        );
        assert_eq!(
            SourceKind::from_path(Path::new("lib.rs")),
            SourceKind::Code { language: "rust" }
        );
        assert_eq!(
            SourceKind::from_path(Path::new("paper.PDF")),
            SourceKind::Pdf
        );
        assert_eq!(
            SourceKind::from_path(Path::new("image.jpeg")),
            SourceKind::Image
        );
        assert_eq!(
            SourceKind::from_path(Path::new("archive.zip")),
            SourceKind::Unsupported
        );

        assert_eq!(
            SourceKind::Markdown.content_category(),
            ContentCategory::Text
        );
        assert_eq!(
            SourceKind::Code { language: "rust" }.content_category(),
            ContentCategory::Text
        );
        assert_eq!(
            SourceKind::Unsupported.content_category(),
            ContentCategory::Other
        );
        assert_eq!(SourceKind::Pdf.format_label(), Some("pdf"));
        assert_eq!(
            SourceKind::Csv { delimiter: b'\t' }.format_label(),
            Some("tsv")
        );
        assert_eq!(
            SourceKind::Code { language: "rust" }.format_label(),
            Some("code")
        );
        assert_eq!(SourceKind::Unsupported.format_label(), None);
    }
}
