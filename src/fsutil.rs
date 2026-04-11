//! Small filesystem helpers shared across the CLI.

use anyhow::{Context, Result};
use atomicwrites::{AllowOverwrite, AtomicFile};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Writes a file atomically by flushing a temp file and replacing the destination in place.
pub fn write_atomic(path: &Path, content: impl AsRef<[u8]>) -> Result<()> {
    let parent = parent_dir(path);
    fs::create_dir_all(&parent)
        .with_context(|| format!("create parent directory for {}", path.display()))?;

    let content = content.as_ref();
    AtomicFile::new(path, AllowOverwrite)
        .write(|file| -> io::Result<()> {
            file.write_all(content)?;
            file.sync_all()?;
            Ok(())
        })
        .map_err(io::Error::from)
        .with_context(|| format!("write file atomically: {}", path.display()))?;

    sync_parent_dir(&parent)
        .with_context(|| format!("sync parent directory for {}", path.display()))?;

    Ok(())
}

fn parent_dir(path: &Path) -> PathBuf {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

#[cfg(unix)]
fn sync_parent_dir(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_dir(_parent: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::{Mutex, OnceLock};

    fn cwd_lock() -> &'static Mutex<()> {
        static CWD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        CWD_LOCK.get_or_init(|| Mutex::new(()))
    }

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

    #[test]
    fn test_write_atomic_supports_paths_without_an_explicit_parent() {
        let _guard = cwd_lock().lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let previous_cwd = env::current_dir().unwrap();
        env::set_current_dir(dir.path()).unwrap();

        write_atomic(Path::new("config.toml"), b"answer = 42\n").unwrap();
        assert_eq!(fs::read_to_string("config.toml").unwrap(), "answer = 42\n");

        env::set_current_dir(previous_cwd).unwrap();
    }
}
