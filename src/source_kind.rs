//! Shared source-kind classification for ingestion and store statistics.

use std::path::Path;

/// File kind recognized by `ragcli`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Text,
    Markdown,
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
        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        if ["md", "markdown"]
            .iter()
            .any(|candidate| ext.eq_ignore_ascii_case(candidate))
        {
            Self::Markdown
        } else if ["txt", "rst"]
            .iter()
            .any(|candidate| ext.eq_ignore_ascii_case(candidate))
        {
            Self::Text
        } else if ext.eq_ignore_ascii_case("pdf") {
            Self::Pdf
        } else if ["png", "jpg", "jpeg", "webp"]
            .iter()
            .any(|candidate| ext.eq_ignore_ascii_case(candidate))
        {
            Self::Image
        } else {
            Self::Unsupported
        }
    }

    /// Returns the metadata format label used for chunk metadata.
    pub fn format_label(self) -> Option<&'static str> {
        match self {
            Self::Text => Some("text"),
            Self::Markdown => Some("markdown"),
            Self::Pdf => Some("pdf"),
            Self::Image => Some("image"),
            Self::Unsupported => None,
        }
    }

    /// Returns the aggregate content category used in store statistics.
    pub fn content_category(self) -> ContentCategory {
        match self {
            Self::Text | Self::Markdown => ContentCategory::Text,
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
            SourceKind::Unsupported.content_category(),
            ContentCategory::Other
        );
        assert_eq!(SourceKind::Pdf.format_label(), Some("pdf"));
        assert_eq!(SourceKind::Unsupported.format_label(), None);
    }
}
