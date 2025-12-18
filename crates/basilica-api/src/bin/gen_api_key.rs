use clap::Parser;

/// Generate a dev API key and print an INSERT statement for Postgres.
///
/// Example:
///   cargo run -p basilica-api --bin gen-api-key -- --user test --name dev \
///     --scopes rentals:* --scopes jobs:*
#[derive(Parser, Debug)]
#[command(
    name = "gen-api-key",
    about = "Generate a dev API key + SQL INSERT for Postgres"
)]
struct Args {
    /// User ID the key belongs to (maps to namespace via u-<user>)
    #[arg(short = 'u', long = "user", default_value = "test")]
    user_id: String,

    /// Human-friendly name, unique per user
    #[arg(short = 'n', long = "name", default_value = "dev")]
    name: String,

    /// Scopes (repeatable), e.g. rentals:* jobs:* training:*
    #[arg(long = "scopes", num_args = 0.., default_values = ["rentals:*", "jobs:*", "training:*"])]
    scopes: Vec<String>,
}

fn escape_sql(s: &str) -> String {
    s.replace("'", "''")
}

fn main() {
    let args = Args::parse();

    // Generate key material
    let generated = basilica_api::api::auth::api_keys::GeneratedApiKey::generate();
    let kid = generated.kid_hex();
    let hash = generated.hash_string().expect("hash");
    let token = generated.into_token();
    let display_token = token.to_string();

    // Build SQL for Postgres
    // scopes TEXT[] literal: ARRAY['a','b']
    let scopes_sql = if args.scopes.is_empty() {
        "ARRAY[]::TEXT[]".to_string()
    } else {
        let items: Vec<String> = args
            .scopes
            .iter()
            .map(|s| format!("'{}'", escape_sql(s)))
            .collect();
        format!("ARRAY[{}]", items.join(","))
    };

    let sql = format!(
        "INSERT INTO api_keys (user_id, kid, name, hash, scopes) VALUES ('{}','{}','{}','{}', {});",
        escape_sql(&args.user_id),
        kid,
        escape_sql(&args.name),
        escape_sql(&hash),
        scopes_sql
    );

    println!("=== Basilica API Key (dev) ===");
    println!("User ID:              {}", args.user_id);
    println!("Name:                 {}", args.name);
    println!("Kid (hex):            {}", kid);
    println!("Token (Authorization): Bearer {}", display_token);
    println!();
    println!(
        "-- Run this against Postgres (compose: postgres service)\n{}",
        sql
    );
}
