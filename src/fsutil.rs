//! Small filesystem helpers shared across the CLI.

use anyhow::{Context, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

/// Writes a file by flushing a uniquely named sibling temp file and then renaming it in place.
pub fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("resolve parent directory for {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create parent directory for {}", path.display()))?;

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time before unix epoch")?
        .as_nanos();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("resolve file name for {}", path.display()))?;
    let tmp_path = parent.join(format!(".{}.{}.{}.tmp", file_name, process::id(), nonce));

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)
        .with_context(|| format!("create temp file for {}", path.display()))?;

    let write_result = (|| -> Result<()> {
        file.write_all(content.as_bytes())
            .with_context(|| format!("write temp file for {}", path.display()))?;
        file.sync_all()
            .with_context(|| format!("flush temp file for {}", path.display()))?;
        Ok(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "rename temp file into place for {}",
            path.display()
        )
    })?;
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
