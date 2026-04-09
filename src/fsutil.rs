//! Small filesystem helpers shared across the CLI.

use anyhow::{Context, Result};
use atomicwrites::{AllowOverwrite, AtomicFile};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Writes a file atomically by flushing a temp file and replacing the destination in place.
pub fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("resolve parent directory for {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create parent directory for {}", path.display()))?;

    AtomicFile::new(path, AllowOverwrite)
        .write(|file| -> io::Result<()> {
            file.write_all(content.as_bytes())?;
            Ok(())
        })
        .map_err(io::Error::from)
        .with_context(|| format!("write file atomically: {}", path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_atomic_creates_and_replaces_file_without_temp_leftovers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        write_atomic(&path, "first = true\n").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "first = true\n");

        write_atomic(&path, "second = true\n").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second = true\n");

        let entries = fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(entries, vec!["config.toml"]);
    }
}
