//! Basilica agent skills installer.

use crate::cli::commands::{SkillsAction, SkillsCommand};
use crate::error::CliError;
use crate::interactive::gate::{ask_multiselect, current, Interactivity};
use crate::progress::{complete_spinner_error, create_spinner};
use clap::CommandFactory;
use color_eyre::eyre::eyre;
use console::style;
use etcetera::{choose_base_strategy, BaseStrategy};
use flate2::read::GzDecoder;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

const DEFAULT_TARBALL_URL: &str =
    "https://github.com/itzlambda/basilica-skills/archive/refs/heads/main.tar.gz";
const TARBALL_URL_ENV: &str = "BASILICA_SKILLS_TARBALL_URL";
const SKILLS_DIR: &str = "skills";
const CURATED_SKILLS: &[&str] = &["basilica-cli"];

#[derive(Clone, Debug, PartialEq, Eq)]
struct CodingTool {
    slug: &'static str,
    name: &'static str,
    aliases: &'static [&'static str],
    label_suffix: Option<&'static str>,
    parent: PathBuf,
    skills_dir_name: &'static str,
    always_default: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InstallTarget {
    tool_name: String,
    skills_dir: PathBuf,
}

type SkillFiles = BTreeMap<String, Vec<(PathBuf, Vec<u8>)>>;

pub async fn handle_skills(cmd: SkillsCommand) -> Result<(), CliError> {
    match cmd.action {
        Some(SkillsAction::Install) => install_skills(&cmd.agent, cmd.yes).await,
        Some(SkillsAction::Uninstall) => uninstall_skills(&cmd.agent, cmd.yes),
        Some(SkillsAction::List) => list_skills(&cmd.agent),
        None => {
            let mut clap = crate::cli::commands::SkillsCommand::command();
            clap.print_help()
                .map_err(|e| CliError::Internal(eyre!("Failed to print help: {}", e)))?;
            println!();
            Ok(())
        }
    }
}

fn tarball_url() -> String {
    std::env::var(TARBALL_URL_ENV).unwrap_or_else(|_| DEFAULT_TARBALL_URL.to_string())
}

fn home_dir() -> Result<PathBuf, CliError> {
    let strategy = choose_base_strategy()
        .map_err(|e| CliError::Internal(eyre!("Failed to determine home directory: {}", e)))?;
    Ok(strategy.home_dir().to_path_buf())
}

fn coding_tools(home: &Path) -> Vec<CodingTool> {
    vec![
        CodingTool {
            slug: "universal",
            name: "Universal (.agents)",
            aliases: &[
                "agents",
                "codex",
                "openai-codex",
                "cursor",
                "opencode",
                "open-code",
                "amp",
                "gemini",
                "gemini-cli",
            ],
            label_suffix: Some("Codex, Cursor, OpenCode, Amp, Gemini"),
            parent: home.join(".agents"),
            skills_dir_name: "skills",
            always_default: true,
        },
        CodingTool {
            slug: "claude-code",
            name: "Claude Code",
            aliases: &["claude"],
            label_suffix: None,
            parent: home.join(".claude"),
            skills_dir_name: "skills",
            always_default: false,
        },
    ]
}

fn matches_agent(tool: &CodingTool, slug: &str) -> bool {
    tool.slug == slug || tool.aliases.contains(&slug)
}

fn resolve_tools(home: &Path, agent_filter: &[String]) -> Result<Vec<CodingTool>, CliError> {
    let all = coding_tools(home);

    if agent_filter.is_empty() {
        return Ok(all
            .into_iter()
            .filter(|tool| tool.always_default || tool.parent.is_dir())
            .collect());
    }

    let mut selected: Vec<CodingTool> = Vec::new();
    for slug in agent_filter {
        match all.iter().find(|tool| matches_agent(tool, slug.as_str())) {
            Some(tool) if !selected.iter().any(|selected| selected.slug == tool.slug) => {
                selected.push(tool.clone());
            }
            Some(_) => {}
            None => {
                let valid = all
                    .iter()
                    .flat_map(|tool| std::iter::once(tool.slug).chain(tool.aliases.iter().copied()))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(CliError::Internal(eyre!(
                    "Unknown agent: '{}'\n\nValid agents: {}",
                    slug,
                    valid
                )));
            }
        }
    }
    Ok(selected)
}

fn default_tool_selections(tools: &[CodingTool]) -> Vec<bool> {
    tools
        .iter()
        .map(|tool| tool.always_default || tool.parent.is_dir())
        .collect()
}

fn select_tools_interactively(
    home: &Path,
    prompt: &str,
    hint: &str,
) -> Result<Vec<CodingTool>, CliError> {
    let tools = coding_tools(home);
    let defaults = default_tool_selections(&tools);
    let labels = tools
        .iter()
        .map(|tool| match tool.label_suffix {
            Some(suffix) => format!("{} - {}", tool.name, suffix),
            None => tool.name.to_string(),
        })
        .collect::<Vec<_>>();
    let selections = ask_multiselect("agents", prompt, &labels, &defaults, hint)?;
    Ok(selections
        .into_iter()
        .filter_map(|idx| tools.get(idx).cloned())
        .collect())
}

fn build_targets(tools: &[CodingTool]) -> Vec<InstallTarget> {
    tools
        .iter()
        .map(|tool| InstallTarget {
            tool_name: tool.name.to_string(),
            skills_dir: tool.parent.join(tool.skills_dir_name),
        })
        .collect()
}

async fn download_tarball() -> Result<Vec<u8>, CliError> {
    let url = tarball_url();
    let response = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .map_err(|e| CliError::Internal(eyre!("Failed to download Basilica skills: {}", e)))?;

    if !response.status().is_success() {
        return Err(CliError::Internal(eyre!(
            "Failed to download Basilica skills from {}: HTTP {}",
            url,
            response.status()
        )));
    }

    response
        .bytes()
        .await
        .map(|bytes| bytes.to_vec())
        .map_err(|e| CliError::Internal(eyre!("Failed to read Basilica skills archive: {}", e)))
}

fn extract_skill_files(tarball_bytes: &[u8]) -> Result<SkillFiles, CliError> {
    let decoder = GzDecoder::new(Cursor::new(tarball_bytes));
    let mut archive = tar::Archive::new(decoder);
    let mut skills: SkillFiles = BTreeMap::new();

    for entry in archive
        .entries()
        .map_err(|e| CliError::Internal(eyre!("Failed to read skills archive: {}", e)))?
    {
        let mut entry =
            entry.map_err(|e| CliError::Internal(eyre!("Failed to read archive entry: {}", e)))?;
        if entry.header().entry_type().is_dir() {
            continue;
        }

        let path = entry
            .path()
            .map_err(|e| CliError::Internal(eyre!("Failed to read archive path: {}", e)))?
            .into_owned();
        let mut components = path.components();

        // GitHub archives have a top-level directory such as basilica-skills-main/.
        let _archive_root = components.next();
        let Some(skills_dir_component) = components.next() else {
            continue;
        };
        if skills_dir_component.as_os_str() != SKILLS_DIR {
            continue;
        }
        let Some(skill_component) = components.next() else {
            continue;
        };
        let skill_name = skill_component.as_os_str().to_string_lossy().to_string();
        let rel_path: PathBuf = components.collect();
        if skill_name.is_empty() || rel_path.as_os_str().is_empty() {
            continue;
        }
        if rel_path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            continue;
        }

        let mut contents = Vec::new();
        entry
            .read_to_end(&mut contents)
            .map_err(|e| CliError::Internal(eyre!("Failed to read archive file: {}", e)))?;
        skills
            .entry(skill_name)
            .or_default()
            .push((rel_path, contents));
    }

    skills.retain(|_, files| files.iter().any(|(path, _)| path == Path::new("SKILL.md")));
    if skills.is_empty() {
        return Err(CliError::Internal(eyre!(
            "No installable Basilica skills found in archive"
        )));
    }
    Ok(skills)
}

fn curated_skill_files(skills: SkillFiles) -> SkillFiles {
    let curated: BTreeSet<&str> = CURATED_SKILLS.iter().copied().collect();
    skills
        .into_iter()
        .filter(|(name, _)| curated.contains(name.as_str()))
        .collect()
}

fn join_rel(dir: &Path, rel: &Path) -> PathBuf {
    let mut path = dir.to_path_buf();
    for component in rel.components() {
        if let std::path::Component::Normal(part) = component {
            path.push(part);
        }
    }
    path
}

fn install_files(targets: &[InstallTarget], skills: &SkillFiles) -> Result<usize, CliError> {
    let mut installed = 0;
    for target in targets {
        std::fs::create_dir_all(&target.skills_dir).map_err(|e| {
            CliError::Internal(eyre!(
                "Failed to create skills directory {}: {}",
                target.skills_dir.display(),
                e
            ))
        })?;

        for (skill_name, files) in skills {
            let skill_dir = target.skills_dir.join(skill_name);
            if skill_dir.exists() {
                std::fs::remove_dir_all(&skill_dir).map_err(|e| {
                    CliError::Internal(eyre!(
                        "Failed to replace existing skill {}: {}",
                        skill_dir.display(),
                        e
                    ))
                })?;
            }

            for (rel, contents) in files {
                let path = join_rel(&skill_dir, rel);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        CliError::Internal(eyre!(
                            "Failed to create directory {}: {}",
                            parent.display(),
                            e
                        ))
                    })?;
                }
                std::fs::write(&path, contents).map_err(|e| {
                    CliError::Internal(eyre!("Failed to write {}: {}", path.display(), e))
                })?;
            }

            installed += 1;
            println!(
                "{} Installed {} to {} -> {}",
                style("✓").green(),
                style(skill_name).green(),
                target.tool_name,
                skill_dir.display()
            );
        }
    }
    Ok(installed)
}

async fn install_skills(agent_filter: &[String], yes: bool) -> Result<(), CliError> {
    let home = home_dir()?;
    let tools = if agent_filter.is_empty() && !yes {
        if matches!(current(), Interactivity::NonInteractive) {
            return Err(CliError::MissingInput {
                field: "agents".to_string(),
                hint: "Re-run with `basilica skills install -y` or pass one or more `--agent <agent>` values.".to_string(),
            });
        }
        select_tools_interactively(
            &home,
            "Select agent targets for Basilica skills",
            "Re-run with `basilica skills install -y` or pass one or more `--agent <agent>` values.",
        )?
    } else {
        resolve_tools(&home, agent_filter)?
    };
    let targets = build_targets(&tools);
    if targets.is_empty() {
        println!("No agent targets selected. Use --agent to choose a target.");
        return Ok(());
    }

    let spinner = create_spinner("Downloading Basilica skills");
    let tarball = match download_tarball().await {
        Ok(tarball) => {
            spinner.finish_and_clear();
            println!("{} Downloaded Basilica skills", style("✓").green());
            tarball
        }
        Err(err) => {
            complete_spinner_error(spinner, "Failed to download Basilica skills");
            return Err(err);
        }
    };
    let skills = curated_skill_files(extract_skill_files(&tarball)?);
    if skills.is_empty() {
        return Err(CliError::Internal(eyre!(
            "No curated Basilica skills found in archive"
        )));
    }

    let installed = install_files(&targets, &skills)?;
    println!(
        "\n{} Installed {} target{}. Restart your agent tool to load new skills.",
        style("✓").green().bold(),
        installed,
        plural_suffix(installed)
    );
    Ok(())
}

fn uninstall_skills(agent_filter: &[String], yes: bool) -> Result<(), CliError> {
    let home = home_dir()?;
    let tools = if agent_filter.is_empty() && !yes {
        if matches!(current(), Interactivity::NonInteractive) {
            return Err(CliError::MissingInput {
                field: "agents".to_string(),
                hint: "Re-run with `basilica skills uninstall -y` or pass one or more `--agent <agent>` values.".to_string(),
            });
        }
        select_tools_interactively(
            &home,
            "Select agent targets to uninstall Basilica skills from",
            "Re-run with `basilica skills uninstall -y` or pass one or more `--agent <agent>` values.",
        )?
    } else {
        resolve_tools(&home, agent_filter)?
    };
    let targets = build_targets(&tools);
    if targets.is_empty() {
        println!("No agent targets selected. Use --agent to choose a target.");
        return Ok(());
    }

    let mut removed = 0;
    for target in &targets {
        for skill in CURATED_SKILLS {
            let skill_dir = target.skills_dir.join(skill);
            match std::fs::remove_dir_all(&skill_dir) {
                Ok(()) => {
                    removed += 1;
                    println!(
                        "{} Removed {} from {} -> {}",
                        style("✓").green(),
                        style(skill).red(),
                        target.tool_name,
                        skill_dir.display()
                    );
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    println!(
                        "{} {}: {} not installed -> {}",
                        style("-").dim(),
                        target.tool_name,
                        skill,
                        skill_dir.display()
                    );
                }
                Err(e) => {
                    return Err(CliError::Internal(eyre!(
                        "Failed to remove {}: {}",
                        skill_dir.display(),
                        e
                    )));
                }
            }
        }
    }
    println!(
        "\n{} Removed {} target{}.",
        style("✓").green().bold(),
        removed,
        plural_suffix(removed)
    );
    Ok(())
}

fn list_skills(agent_filter: &[String]) -> Result<(), CliError> {
    let home = home_dir()?;
    let tools = resolve_tools(&home, agent_filter)?;
    let targets = build_targets(&tools);

    println!("Available skills:");
    for skill in CURATED_SKILLS {
        println!("  {}", style(skill).cyan());
    }

    println!("\nTargets:");
    if targets.is_empty() {
        println!("  none detected");
        return Ok(());
    }

    for target in &targets {
        let installed = CURATED_SKILLS
            .iter()
            .filter(|skill| target.skills_dir.join(skill).join("SKILL.md").is_file())
            .copied()
            .collect::<Vec<_>>();
        let installed_text = if installed.is_empty() {
            "none".to_string()
        } else {
            installed.join(", ")
        };
        println!(
            "  {} -> {} ({})",
            target.tool_name,
            target.skills_dir.display(),
            installed_text
        );
    }
    Ok(())
}

fn plural_suffix(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use tar::{Builder, Header};

    fn archive(files: &[(&str, &[u8])]) -> Vec<u8> {
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut builder = Builder::new(encoder);
        for (path, contents) in files {
            let mut header = Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, *path, &mut &contents[..])
                .unwrap();
        }
        let encoder = builder.into_inner().unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn resolves_default_targets_with_universal_always_present() {
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join(".codex")).unwrap();

        let tools = resolve_tools(home.path(), &[]).unwrap();
        let slugs = tools.iter().map(|tool| tool.slug).collect::<Vec<_>>();
        assert_eq!(slugs, vec!["universal"]);
    }

    #[test]
    fn resolves_codex_agent_to_universal_target() {
        let home = tempfile::tempdir().unwrap();

        let tools = resolve_tools(home.path(), &[String::from("codex")]).unwrap();
        let targets = build_targets(&tools);

        assert_eq!(
            tools.iter().map(|tool| tool.slug).collect::<Vec<_>>(),
            vec!["universal"]
        );
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].skills_dir,
            home.path().join(".agents").join("skills")
        );
    }

    #[test]
    fn resolves_agent_skills_compatible_agents_to_universal_target() {
        let home = tempfile::tempdir().unwrap();

        let tools = resolve_tools(
            home.path(),
            &[
                String::from("cursor"),
                String::from("opencode"),
                String::from("amp"),
                String::from("gemini-cli"),
            ],
        )
        .unwrap();

        assert_eq!(
            tools.iter().map(|tool| tool.slug).collect::<Vec<_>>(),
            vec!["universal"]
        );
    }

    #[test]
    fn rejects_unknown_agent_filter() {
        let home = tempfile::tempdir().unwrap();
        let err = resolve_tools(home.path(), &[String::from("unknown")]).unwrap_err();
        assert!(format!("{err:?}").contains("Unknown agent"));
    }

    #[test]
    fn extracts_skill_directories_under_skills_root() {
        let tarball = archive(&[
            ("basilica-skills-main/README.md", b"root"),
            (
                "basilica-skills-main/skills/basilica-cli/SKILL.md",
                b"skill",
            ),
            (
                "basilica-skills-main/skills/basilica-cli/notes.md",
                b"notes",
            ),
            ("basilica-skills-main/not-a-skill/file.txt", b"ignored"),
            ("basilica-skills-main/other-skill/SKILL.md", b"ignored"),
        ]);

        let skills = extract_skill_files(&tarball).unwrap();
        assert!(skills.contains_key("basilica-cli"));
        assert!(!skills.contains_key("not-a-skill"));
        assert_eq!(skills["basilica-cli"].len(), 2);
    }

    #[test]
    fn curated_skills_ignore_unknown_archive_entries() {
        let mut skills = SkillFiles::new();
        skills.insert(
            "basilica-cli".to_string(),
            vec![(PathBuf::from("SKILL.md"), b"cli".to_vec())],
        );
        skills.insert(
            "other".to_string(),
            vec![(PathBuf::from("SKILL.md"), b"other".to_vec())],
        );

        let curated = curated_skill_files(skills);
        assert_eq!(curated.keys().collect::<Vec<_>>(), vec!["basilica-cli"]);
    }

    #[test]
    fn install_files_replaces_curated_skill_without_touching_others() {
        let root = tempfile::tempdir().unwrap();
        let target = InstallTarget {
            tool_name: "Test".to_string(),
            skills_dir: root.path().join("skills"),
        };
        std::fs::create_dir_all(target.skills_dir.join("other")).unwrap();
        std::fs::write(target.skills_dir.join("other").join("SKILL.md"), "other").unwrap();
        std::fs::create_dir_all(target.skills_dir.join("basilica-cli")).unwrap();
        std::fs::write(
            target.skills_dir.join("basilica-cli").join("old.txt"),
            "old",
        )
        .unwrap();

        let mut skills = SkillFiles::new();
        skills.insert(
            "basilica-cli".to_string(),
            vec![(PathBuf::from("SKILL.md"), b"new".to_vec())],
        );

        install_files(std::slice::from_ref(&target), &skills).unwrap();
        assert_eq!(
            std::fs::read_to_string(target.skills_dir.join("basilica-cli").join("SKILL.md"))
                .unwrap(),
            "new"
        );
        assert!(!target
            .skills_dir
            .join("basilica-cli")
            .join("old.txt")
            .exists());
        assert!(target.skills_dir.join("other").join("SKILL.md").exists());
    }
}
