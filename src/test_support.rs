#[cfg(test)]
use crate::config;
#[cfg(test)]
use std::env;
#[cfg(test)]
use std::io::{Read, Write};
#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use std::thread;

#[cfg(test)]
pub async fn with_test_env<T, F>(
    config_home: &Path,
    ollama_url: Option<&str>,
    f: impl FnOnce() -> F,
) -> T
where
    F: std::future::Future<Output = T>,
{
    let _guard = config::test_env_lock().lock().unwrap();
    let previous_xdg = env::var_os("XDG_CONFIG_HOME");
    let previous_ollama = env::var_os(config::ENV_OLLAMA_URL);

    unsafe {
        env::set_var("XDG_CONFIG_HOME", config_home);
        match ollama_url {
            Some(url) => env::set_var(config::ENV_OLLAMA_URL, url),
            None => env::remove_var(config::ENV_OLLAMA_URL),
        }
    }

    let result = f().await;

    unsafe {
        match previous_xdg {
            Some(value) => env::set_var("XDG_CONFIG_HOME", value),
            None => env::remove_var("XDG_CONFIG_HOME"),
        }
        match previous_ollama {
            Some(value) => env::set_var(config::ENV_OLLAMA_URL, value),
            None => env::remove_var(config::ENV_OLLAMA_URL),
        }
    }

    result
}

#[cfg(test)]
pub fn sequential_json_server(bodies: Vec<&'static str>) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().unwrap();

    thread::spawn(move || {
        for body in bodies {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        }
    });

    format!("http://{}", addr)
}
